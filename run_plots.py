import os, re, sys, select, runpy, traceback, atexit

os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(os.getcwd(), "plots")
os.makedirs(SAVE_DIR, exist_ok=True)

try:
    import termios, tty

    _fd = sys.stdin.fileno()
    _old = termios.tcgetattr(_fd)
    tty.setcbreak(_fd)
    atexit.register(lambda: termios.tcsetattr(_fd, termios.TCSADRAIN, _old))
except Exception:
    pass

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "QtAgg"))
import matplotlib.pyplot as plt
import matplotlib.figure as mf

matplotlib.use = lambda *a, **k: None
plt.rcParams["keymap.save"] = []
plt.rcParams["savefig.directory"] = SAVE_DIR
real_savefig, real_close = mf.Figure.savefig, plt.close
counter = [0]


def fig_name(fig, n):
    t = (
        fig._suptitle.get_text()
        if fig._suptitle
        else (
            fig.axes[0].get_title(loc="left") or fig.axes[0].get_title()
            if fig.axes
            else ""
        )
    )
    t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")
    return os.path.join(SAVE_DIR, f"{n:02d}_{t or 'plot'}.png")


def save_png(fig, n):
    try:
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = fig_name(fig, n)
        real_savefig(fig, path, dpi=200, bbox_inches="tight")
        print("Saved:", path, flush=True)
    except Exception:
        traceback.print_exc()
        return
    try:
        fig.canvas.manager.set_window_title("Saved: " + path)
    except Exception:
        pass


def present(fig):
    counter[0] += 1
    n = counter[0]
    try:
        fig.set_size_inches(16, 9)
        fig.tight_layout()
    except Exception:
        pass
    fig.canvas.mpl_connect(
        "key_press_event", lambda e: save_png(fig, n) if e.key == "s" else None
    )
    plt.figure(fig.number)
    plt.show(block=False)
    try:
        fig.canvas.manager.window.showMaximized()
    except Exception:
        pass
    plt.pause(0.2)
    while plt.fignum_exists(fig.number):
        print(
            f"\n[Plot {n}] Press: q = next  | s = save PNG | f = full screen",
            flush=True,
        )
        while not select.select([sys.stdin], [], [], 0)[0]:
            plt.pause(0.1)
            if not plt.fignum_exists(fig.number):
                return
        cmd = os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()
        if cmd == "f":
            try:
                fig.canvas.manager.full_screen_toggle()
            except Exception:
                pass
            continue
        if cmd == "s":
            save_png(fig, n)
            continue
        if cmd == "x":
            real_close("all")
            sys.exit(0)
        break
    real_close(fig)


mf.Figure.savefig = lambda self, *a, **k: present(self)
plt.savefig = lambda *a, **k: present(plt.gcf())
plt.show = lambda *a, **k: None

try:
    runpy.run_path("my_python_analysessupplement.py", run_name="__main__")
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
print("\nDone..")
