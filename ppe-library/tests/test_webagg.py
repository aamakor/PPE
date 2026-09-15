"""Set PPE_WEBAGG_NETWORK_TESTS=1 to include real localhost/WebSocket checks."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import matplotlib
from ppe.plotting import NavigationPlot


class WebAggUnitTests(unittest.TestCase):
    def test_webagg_uses_server_pump_instead_of_blocking_show(self):
        previous = matplotlib.get_backend()
        try:
            with patch("ppe._webagg.WebAggServer") as server:
                plot = NavigationPlot(live=True, backend="WebAgg", webagg_port=9123)
                try:
                    server.assert_called_once_with(plot.figure, 9123)
                    with patch.object(plot.plt, "show") as show:
                        plot.on_initial({"validation": [1, 2, 3], "test": [2, 3, 4]})
                        show.assert_not_called()
                        server.return_value.pump.assert_called()
                    with patch.object(plot, "read_input", return_value="") as read:
                        plot.show()
                        read.assert_called_once()
                finally:
                    plot.close()
                server.return_value.close.assert_called_once()
        finally:
            matplotlib.use(previous)


@unittest.skipUnless(os.environ.get("PPE_WEBAGG_NETWORK_TESTS") == "1", "Localhost socket tests are opt-in")
class WebAggNetworkTests(unittest.TestCase):
    def test_http_websocket_updates_and_port_cleanup(self):
        from tornado.httpclient import AsyncHTTPClient
        from tornado.websocket import websocket_connect
        previous = matplotlib.get_backend()
        plot = NavigationPlot(live=True, backend="WebAgg", webagg_port=0)
        port = plot._web_server.port
        try:
            plot.on_initial({"validation": [1, 2, 3], "test": [2, 3, 4]})
            loop = plot._web_server.loop

            async def connect():
                client = AsyncHTTPClient()
                page = await client.fetch(plot.url)
                self.assertEqual(page.code, 200)
                self.assertIn(b"mpl.figure", page.body)
                image = await client.fetch(plot.url + "/download.png")
                self.assertTrue(image.body.startswith(b"\x89PNG"))
                return await websocket_connect(plot.url.replace("http://", "ws://") + "/ws")

            socket = loop.run_sync(connect, timeout=10)

            async def frame():
                await socket.write_message(json.dumps({"type": "draw"}))
                for _ in range(20):
                    message = await socket.read_message()
                    if isinstance(message, bytes):
                        self.assertTrue(message.startswith(b"\x89PNG"))
                        return message
                self.fail("No binary image received from WebAgg")

            initial_frame = loop.run_sync(frame, timeout=10)
            plot.on_cycle({"cycle": 1, "preference": [-1, 0, 0], "predictor": [0.8, 2.2, 3.1],
                           "corrector": [0.7, 2.1, 3.0]})
            next_frame = loop.run_sync(frame, timeout=10)
            self.assertNotEqual(initial_frame, next_frame)
            socket.close()
        finally:
            plot.close()
            matplotlib.use(previous)
        # Closing a plot releases its listening port for another plot.
        again = NavigationPlot(live=True, backend="WebAgg", webagg_port=port)
        again.close()
        matplotlib.use(previous)

    def test_interactive_cli_completes_a_cycle_and_exits(self):
        environment = os.environ.copy()
        environment.pop("DISPLAY", None)
        environment.pop("WAYLAND_DISPLAY", None)
        for backend_flags in (["--plot-backend", "WebAgg"], []):
            with self.subTest(backend_flags=backend_flags):
                with tempfile.TemporaryDirectory() as folder:
                    run = subprocess.run([sys.executable, "-m", "ppe.cli", "demo", "--interactive", "--plot",
                                          *backend_flags, "--webagg-port", "0", "--output", folder],
                                         input="1\n1\nq\n\n", capture_output=True, text=True, timeout=60, env=environment)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    self.assertIn("WebAgg plot: http://127.0.0.1:", run.stdout)
                    self.assertEqual(run.stdout.count("Select objectives"), 2)
                    record = json.loads((Path(folder) / "results/navigation.json").read_text())
                    self.assertEqual(record["status"], "stopped")
                    self.assertEqual(len(record["cycles"]), 1)
                    self.assertTrue((Path(folder) / "results/navigation-3d.png").exists())
                    if not backend_flags:
                        self.assertIn("QtAgg and TkAgg are unavailable; using WebAgg", run.stdout)


if __name__ == "__main__":
    unittest.main()
