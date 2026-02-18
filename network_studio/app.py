import os
import pickle
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict, List

import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def load_graph_pickle(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, nx.Graph):
        raise TypeError(f"Pickle does not contain a networkx.Graph. Got: {type(obj)}")
    # Ensure positions exist
    for n in obj.nodes:
        if "position" not in obj.nodes[n]:
            raise ValueError(f'Node {n} missing "position" attribute.')
    return obj


def save_graph_pickle(G: nx.Graph, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(G, f)


def get_pos(G: nx.Graph, n: Any) -> np.ndarray:
    p = G.nodes[n]["position"]
    return np.array(p, dtype=float)


def set_pos(G: nx.Graph, n: Any, p: np.ndarray) -> None:
    G.nodes[n]["position"] = np.array([float(p[0]), float(p[1])])


def dist2(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(d[0] * d[0] + d[1] * d[1])


def point_to_segment_distance2(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    # squared distance from p to segment a-b
    ab = b - a
    ap = p - a
    denom = float(ab[0] * ab[0] + ab[1] * ab[1])
    if denom == 0.0:
        return dist2(p, a)
    t = float((ap[0] * ab[0] + ap[1] * ab[1]) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return dist2(p, proj)

def project_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Project point p onto segment a-b.
    Returns (projection_point, t) where proj = a + t*(b-a), t in [0,1].
    """
    ab = b - a
    denom = float(ab[0]*ab[0] + ab[1]*ab[1])
    if denom == 0.0:
        return a.copy(), 0.0
    t = float(((p - a)[0]*ab[0] + (p - a)[1]*ab[1]) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return proj, t

@dataclass
class Hit:
    kind: str  # "node" or "edge"
    node: Any = None
    edge: Tuple[Any, Any] = None


class NetworkStudioApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Network Studio (nx.Graph editor)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.G: nx.Graph = nx.Graph()
        self.path: Optional[str] = None

        self.mode = "MOVE"  # MOVE / ADD_NODE / ADD_EDGE / DELETE
        self.status = tk.StringVar(value="Mode: MOVE (M) | Load (Ctrl+O) | Save (Ctrl+S)")

        self.selected_node_for_edge: Optional[Any] = None
        self.dragging_node: Optional[Any] = None
        self.drag_offset = np.zeros(2)

        self.undo_stack: List[bytes] = []  # store pickled graphs for simple undo

        # --- Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # --- Status bar
        status_bar = tk.Label(root, textvariable=self.status, anchor="w")
        status_bar.pack(fill=tk.X)

        # --- Menu
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open... (Ctrl+O)", command=self.open_file)
        filemenu.add_command(label="Save (Ctrl+S)", command=self.save_file)
        filemenu.add_command(label="Save As...", command=self.save_as)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Controls", command=self.show_controls)
        menubar.add_cascade(label="Help", menu=helpmenu)
        root.config(menu=menubar)

        # --- Bindings
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        root.bind("<Control-o>", lambda e: self.open_file())
        root.bind("<Control-s>", lambda e: self.save_file())

        root.bind("m", lambda e: self.set_mode("MOVE"))
        root.bind("n", lambda e: self.set_mode("ADD_NODE"))
        root.bind("e", lambda e: self.set_mode("ADD_EDGE"))
        root.bind("d", lambda e: self.set_mode("DELETE"))
        root.bind("r", lambda e: self.rescale_view())
        root.bind("u", lambda e: self.undo())
        root.bind("a", lambda e: self.set_mode("SPLIT_EDGE"))

        # start with a blank graph
        self.rescale_view()
        self.redraw()

    # ---------- UI helpers
    def push_undo(self):
        try:
            self.undo_stack.append(pickle.dumps(self.G))
            if len(self.undo_stack) > 30:
                self.undo_stack = self.undo_stack[-30:]
        except Exception:
            pass

    def undo(self):
        if not self.undo_stack:
            self.set_status("Undo stack empty.")
            return
        blob = self.undo_stack.pop()
        self.G = pickle.loads(blob)
        self.selected_node_for_edge = None
        self.dragging_node = None
        self.redraw()
        self.set_status("Undid last action.")

    def set_status(self, msg: str):
        self.status.set(f"Mode: {self.mode} | {msg}")

    def set_mode(self, mode: str):
        self.mode = mode
        self.selected_node_for_edge = None
        self.dragging_node = None
        self.set_status("Ready.")
        self.redraw()

    def show_controls(self):
        msg = (
            "Controls:\n"
            "  M : Move nodes (drag)\n"
            "  N : Add node (click)\n"
            "  E : Add edge (click two nodes)\n"
            "  D : Delete node (click node)\n"
            "      Shift + click near an edge: delete edge\n"
            "  A : Add node on edge (split edge)\n"
            "  U : Undo\n"
            "  R : Rescale view\n"
            "  Ctrl+O : Open pickle\n"
            "  Ctrl+S : Save\n"
            "Mouse wheel: zoom\n"
        )
        messagebox.showinfo("Controls", msg)

    # ---------- File IO
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Open nx.Graph pickle",
            filetypes=[("Pickle files", "*.pkl *.pickle"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            G = load_graph_pickle(path)
        except Exception as ex:
            messagebox.showerror("Load error", str(ex))
            return
        self.push_undo()
        self.G = G
        self.path = path
        self.selected_node_for_edge = None
        self.dragging_node = None
        self.rescale_view()
        self.redraw()
        self.set_status(f"Loaded: {os.path.basename(path)}")

    def save_file(self):
        if self.path is None:
            return self.save_as()
        try:
            save_graph_pickle(self.G, self.path)
        except Exception as ex:
            messagebox.showerror("Save error", str(ex))
            return
        self.set_status(f"Saved: {os.path.basename(self.path)}")

    def save_as(self):
        path = filedialog.asksaveasfilename(
            title="Save nx.Graph pickle as",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl *.pickle"), ("All files", "*.*")]
        )
        if not path:
            return
        self.path = path
        self.save_file()

    # ---------- Hit testing
    def hit_test(self, x: float, y: float, *, shift: bool = False, allow_edge: bool = False) -> Optional[Hit]:
        if self.G.number_of_nodes() == 0:
            return None

        p = np.array([x, y], dtype=float)

        # tolerance based on view scale
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        scale = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
        node_tol2 = (0.015 * scale) ** 2
        edge_tol2 = (0.010 * scale) ** 2

        # node hit
        best_n = None
        best_d2 = node_tol2
        for n in self.G.nodes:
            pn = get_pos(self.G, n)
            d2 = dist2(p, pn)
            if d2 < best_d2:
                best_d2 = d2
                best_n = n

        if best_n is not None:
            return Hit(kind="node", node=best_n)

        # edge hit only if shift (to avoid annoying deletions)
        if (shift or allow_edge) and self.G.number_of_edges() > 0:
            best_e = None
            best_ed2 = edge_tol2
            for u, v in self.G.edges:
                pu = get_pos(self.G, u)
                pv = get_pos(self.G, v)
                d2 = point_to_segment_distance2(p, pu, pv)
                if d2 < best_ed2:
                    best_ed2 = d2
                    best_e = (u, v)
            if best_e is not None:
                return Hit(kind="edge", edge=best_e)

        return None

    # ---------- Interaction
    def on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)
        shift = (event.key == "shift")

        if self.mode == "MOVE":
            hit = self.hit_test(x, y, shift=False)
            if hit and hit.kind == "node":
                self.dragging_node = hit.node
                p = get_pos(self.G, hit.node)
                self.drag_offset = p - np.array([x, y])
                self.set_status(f"Dragging node {hit.node}")
            return

        if self.mode == "ADD_NODE":
            self.push_undo()
            new_id = self._next_node_id()
            self.G.add_node(new_id, position=np.array([x, y], dtype=float))
            self.set_status(f"Added node {new_id}")
            self.redraw()
            return

        if self.mode == "ADD_EDGE":
            hit = self.hit_test(x, y, shift=False)
            if not hit or hit.kind != "node":
                self.set_status("Click a node to start/finish an edge.")
                return

            if self.selected_node_for_edge is None:
                self.selected_node_for_edge = hit.node
                self.set_status(f"Selected source node {hit.node}. Click another node.")
                self.redraw()
                return

            # second click
            u = self.selected_node_for_edge
            v = hit.node
            if u == v:
                self.set_status("Same node; pick a different node.")
                return

            self.push_undo()
            self.G.add_edge(u, v)
            self.set_status(f"Added edge ({u}, {v})")
            self.selected_node_for_edge = None
            self.redraw()
            return

        if self.mode == "DELETE":
            hit = self.hit_test(x, y, shift=shift)
            if not hit:
                self.set_status("Nothing to delete here.")
                return

            self.push_undo()
            if hit.kind == "node":
                n = hit.node
                self.G.remove_node(n)
                self.set_status(f"Deleted node {n}")
            elif hit.kind == "edge":
                u, v = hit.edge
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)
                self.set_status(f"Deleted edge ({u}, {v})")
            self.redraw()
            return
        
        if self.mode == "SPLIT_EDGE":
            # click near an edge, insert node exactly on that edge
            hit = self.hit_test(x, y, shift=False, allow_edge=True)
            if not hit or hit.kind != "edge":
                self.set_status("Click near an edge to split it.")
                return

            u, v = hit.edge
            pu = get_pos(self.G, u)
            pv = get_pos(self.G, v)
            click = np.array([x, y], dtype=float)

            proj, t = project_point_to_segment(click, pu, pv)

            # Optional: avoid splitting extremely close to endpoints
            if t < 0.02 or t > 0.98:
                self.set_status("Too close to an endpoint; zoom in and click nearer the middle.")
                return

            self.push_undo()
            new_id = self._next_node_id()
            self.G.add_node(new_id, position=proj)

            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
            self.G.add_edge(u, new_id)
            self.G.add_edge(new_id, v)

            self.set_status(f"Split edge ({u},{v}) with new node {new_id}")
            self.redraw()
            return

    def on_motion(self, event):
        if self.mode != "MOVE":
            return
        if self.dragging_node is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)
        p = np.array([x, y], dtype=float) + self.drag_offset
        set_pos(self.G, self.dragging_node, p)
        self.redraw(keep_limits=True)

    def on_release(self, event):
        if self.mode == "MOVE" and self.dragging_node is not None:
            self.push_undo()
            self.set_status(f"Moved node {self.dragging_node}")
            self.dragging_node = None

    def on_scroll(self, event):
        # zoom around cursor
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        base_scale = 1.15
        if event.button == "up":
            scale_factor = 1 / base_scale
        else:
            scale_factor = base_scale

        x, y = float(event.xdata), float(event.ydata)
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        relx = (x - xlim[0]) / (xlim[1] - xlim[0])
        rely = (y - ylim[0]) / (ylim[1] - ylim[0])

        self.ax.set_xlim([x - relx * new_width, x + (1 - relx) * new_width])
        self.ax.set_ylim([y - rely * new_height, y + (1 - rely) * new_height])
        self.canvas.draw_idle()

    # ---------- Drawing
    def redraw(self, keep_limits: bool = False):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.clear()

        # edges
        for u, v in self.G.edges:
            pu = get_pos(self.G, u)
            pv = get_pos(self.G, v)
            self.ax.plot([pu[0], pv[0]], [pu[1], pv[1]], linewidth=0.8)

        # nodes
        if self.G.number_of_nodes() > 0:
            P = np.array([get_pos(self.G, n) for n in self.G.nodes])
            self.ax.scatter(P[:, 0], P[:, 1], s=12)

        # highlight selected edge-start node
        if self.selected_node_for_edge is not None and self.selected_node_for_edge in self.G:
            p = get_pos(self.G, self.selected_node_for_edge)
            self.ax.scatter([p[0]], [p[1]], s=60)

        self.ax.set_aspect("equal", "box")
        self.ax.set_title(f"Network Studio | Nodes={self.G.number_of_nodes()} Edges={self.G.number_of_edges()}")

        if keep_limits:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def rescale_view(self):
        # fit view to graph
        if self.G.number_of_nodes() == 0:
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.canvas.draw_idle()
            return
        P = np.array([get_pos(self.G, n) for n in self.G.nodes])
        xmin, ymin = P.min(axis=0)
        xmax, ymax = P.max(axis=0)
        dx = xmax - xmin
        dy = ymax - ymin
        pad = 0.15 * max(dx, dy, 1e-6)
        self.ax.set_xlim(xmin - pad, xmax + pad)
        self.ax.set_ylim(ymin - pad, ymax + pad)
        self.canvas.draw_idle()

    def on_close(self):
        """
        Properly shut down Tk + Matplotlib so Python exits cleanly.
        """
        try:
            plt.close(self.fig)   # close matplotlib figure
        except Exception:
            pass

        try:
            self.root.quit()      # stop Tk mainloop
            self.root.destroy()   # destroy window
        except Exception:
            pass

    def _next_node_id(self) -> int:
        # Choose an integer id not currently used
        if self.G.number_of_nodes() == 0:
            return 0
        ints = [n for n in self.G.nodes if isinstance(n, int)]
        if len(ints) != self.G.number_of_nodes():
            # if mixed ids, pick max int + 1 and hope it's unused
            m = max([i for i in ints], default=-1)
            nid = m + 1
            while nid in self.G:
                nid += 1
            return nid
        nid = max(ints) + 1
        while nid in self.G:
            nid += 1
        return nid


def main():
    root = tk.Tk()
    app = NetworkStudioApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
