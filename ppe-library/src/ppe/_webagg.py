"""WebAgg server serviced on the plotting thread between training phases."""
import asyncio


class WebAggServer:
    def __init__(self, figure, port):
        from matplotlib.backends.backend_webagg import WebAggApplication
        from tornado.httpserver import HTTPServer
        from tornado.ioloop import IOLoop
        from tornado.netutil import bind_sockets
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise ValueError("Use PPE WebAgg from a normal Python script or CLI, outside an already running asyncio loop.")
        self.loop = IOLoop.current()
        self.manager = figure.canvas.manager
        self.server = HTTPServer(WebAggApplication())
        sockets = bind_sockets(port, address="127.0.0.1")
        self.port = sockets[0].getsockname()[1]
        try:
            self.server.add_sockets(sockets)
        except Exception:
            for socket in sockets:
                socket.close()
            raise
        self.url = f"http://127.0.0.1:{self.port}/{figure.number}"
        self.closed = False
        print(f"WebAgg plot: {self.url}", flush=True)

    def pump(self, seconds):
        from tornado.gen import sleep
        if not self.closed:
            self.loop.run_sync(lambda: sleep(seconds))

    def close(self):
        if not self.closed:
            self.server.stop()
            for websocket in list(self.manager.web_sockets):
                websocket.close()
            self.loop.run_sync(self.server.close_all_connections)
            self.closed = True
