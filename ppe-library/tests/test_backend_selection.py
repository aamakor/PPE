"""Desktop availability controls whether live plotting starts WebAgg."""
import contextlib
import io
import unittest
from unittest.mock import Mock, call, patch

import matplotlib
from ppe.plotting import NavigationPlot, _create_figure


class BackendSelectionTests(unittest.TestCase):
    def test_usable_qt_does_not_try_tk_or_webagg(self):
        mpl, plt = Mock(), Mock()
        figure = _create_figure(mpl, plt, live=True, backend=None)
        self.assertIs(figure, plt.figure.return_value)
        self.assertEqual(mpl.use.call_args_list, [call("QtAgg")])

    def test_tk_is_used_when_qt_cannot_start(self):
        # The binding can import successfully but fail when opening a window.
        mpl, plt = Mock(), Mock()
        desktop = object()
        plt.figure.side_effect = [RuntimeError("No Qt display"), desktop]
        self.assertIs(_create_figure(mpl, plt, live=True, backend=None), desktop)
        self.assertEqual(mpl.use.call_args_list, [call("QtAgg"), call("TkAgg")])

    def test_both_desktop_failures_start_webagg(self):
        previous = matplotlib.get_backend()
        original_use = matplotlib.use

        def use(backend):
            if backend in ("QtAgg", "TkAgg"):
                raise ImportError("Desktop unavailable")
            original_use(backend)

        try:
            with patch.object(matplotlib, "use", side_effect=use) as select, \
                    patch("ppe._webagg.WebAggServer") as server, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                plot = NavigationPlot(live=True, webagg_port=0)
                try:
                    self.assertTrue(plot.is_webagg)
                    self.assertEqual(select.call_args_list,
                                     [call("QtAgg"), call("TkAgg"), call("WebAgg")])
                    server.assert_called_once_with(plot.figure, 0)
                    self.assertIn("using WebAgg", output.getvalue())
                finally:
                    plot.close()
        finally:
            original_use(previous)

    def test_explicit_backend_failure_does_not_fall_back(self):
        mpl, plt = Mock(), Mock()
        mpl.use.side_effect = ImportError("Requested backend unavailable")
        with self.assertRaisesRegex(ImportError, "Requested backend unavailable"):
            _create_figure(mpl, plt, live=True, backend="TkAgg")
        self.assertEqual(mpl.use.call_args_list, [call("TkAgg")])
        plt.figure.assert_not_called()

    def test_saved_plot_does_not_select_a_live_backend(self):
        mpl, plt = Mock(), Mock()
        _create_figure(mpl, plt, live=False, backend=None)
        mpl.use.assert_not_called()


if __name__ == "__main__":
    unittest.main()
