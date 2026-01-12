import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import networkx as nx

from graphlet.graphlet_counter import (
    enumerate_3node_graphlets,
    enumerate_4node_graphlets,
)


class GraphletViewer(tk.Tk):
    def __init__(self, G, pos, triangles, paths, motifs4):
        super().__init__()
        self.title("Graphlet Viewer")

        self.G = G
        self.pos = pos
        self.triangles = triangles         # list of 3-tuples
        self.paths = paths                 # list of (center, side1, side2)
        self.motifs4 = motifs4             # dict 'g4_1'..'g4_6' -> list of 4-tuples

        # main layout: left = lists, right = plot
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_plot()

        # initial draw
        self.highlight_nodes = None
        self.draw_graph()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_left_panel(self):
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky="ns")

        notebook = ttk.Notebook(left)
        notebook.pack(fill="both", expand=True)

        # ---------- Triangles tab ----------
        tri_frame = ttk.Frame(notebook)
        notebook.add(tri_frame, text="Triangles")

        self.tri_list = tk.Listbox(tri_frame, height=15, width=25)
        self.tri_list.pack(side="left", fill="both", expand=True)

        tri_scroll = ttk.Scrollbar(tri_frame, orient="vertical",
                                   command=self.tri_list.yview)
        tri_scroll.pack(side="right", fill="y")
        self.tri_list.configure(yscrollcommand=tri_scroll.set)

        for i, t in enumerate(self.triangles):
            self.tri_list.insert(tk.END, f"{i}: {t}")

        self.tri_list.bind("<<ListboxSelect>>", self.on_triangle_select)

        # ---------- 3-node Paths tab ----------
        path_frame = ttk.Frame(notebook)
        notebook.add(path_frame, text="3-node Paths")

        self.path_list = tk.Listbox(path_frame, height=15, width=25)
        self.path_list.pack(side="left", fill="both", expand=True)

        path_scroll = ttk.Scrollbar(path_frame, orient="vertical",
                                    command=self.path_list.yview)
        path_scroll.pack(side="right", fill="y")
        self.path_list.configure(yscrollcommand=path_scroll.set)

        for i, p in enumerate(self.paths):
            self.path_list.insert(tk.END, f"{i}: {p}")

        self.path_list.bind("<<ListboxSelect>>", self.on_path_select)

        # ---------- 4-node motifs tabs ----------
        self.g4_lists = {}  # motif_type -> Listbox

        # pretty labels for tabs (optional)
        motif_labels = {
            "g4_1": "g4_1 K4",
            "g4_2": "g4_2 Diamond",
            "g4_3": "g4_3 Tailed tri",
            "g4_4": "g4_4 C4",
            "g4_5": "g4_5 3-star",
            "g4_6": "g4_6 P4",
        }

        for key in ["g4_1", "g4_2", "g4_3", "g4_4", "g4_5", "g4_6"]:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=motif_labels.get(key, key))

            lb = tk.Listbox(frame, height=15, width=25)
            lb.pack(side="left", fill="both", expand=True)

            scroll = ttk.Scrollbar(frame, orient="vertical",
                                   command=lb.yview)
            scroll.pack(side="right", fill="y")
            lb.configure(yscrollcommand=scroll.set)

            quads = self.motifs4.get(key, [])
            for i, q in enumerate(quads):
                lb.insert(tk.END, f"{i}: {q}")

            # bind with a closure capturing key
            lb.bind("<<ListboxSelect>>",
                    lambda event, mkey=key: self.on_g4_select(event, mkey))

            self.g4_lists[key] = lb

    def _build_plot(self):
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def draw_graph(self):
        self.ax.clear()

        # ---------- draw all edges in black ----------
        for u, v in self.G.edges():
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            self.ax.plot([x1, x2], [y1, y2], color="k", linewidth=0.6)

        # ---------- highlight edges of selected graphlet ----------
        if self.highlight_nodes:
            H = self.highlight_nodes
            for u, v in self.G.edges():
                if u in H and v in H:
                    x1, y1 = self.pos[u]
                    x2, y2 = self.pos[v]
                    self.ax.plot([x1, x2], [y1, y2], linewidth=2.5)

        # ---------- draw nodes (grey + highlighted) ----------
        xs = []
        ys = []
        colors = []
        sizes = []
        for u in self.G.nodes():
            x, y = self.pos[u]
            xs.append(x)
            ys.append(y)
            if self.highlight_nodes and u in self.highlight_nodes:
                colors.append("C1")  # highlighted node colour
                sizes.append(120)
            else:
                colors.append("0.7")  # grey
                sizes.append(40)

        self.ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="k", linewidths=0.5)

        # labels (small)
        for u in self.G.nodes():
            x, y = self.pos[u]
            self.ax.text(x, y, str(u), fontsize=8,
                         ha="center", va="center")

        self.ax.set_aspect("equal")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw()

    # ---------- selection helpers ----------

    def clear_all_selections(self, except_widget=None):
        # clear triangles and paths
        if except_widget is not self.tri_list:
            self.tri_list.selection_clear(0, tk.END)
        if except_widget is not self.path_list:
            self.path_list.selection_clear(0, tk.END)
        # clear all g4_* lists
        for lb in self.g4_lists.values():
            if lb is not except_widget:
                lb.selection_clear(0, tk.END)

    # ---------- callbacks for list selections ----------

    def on_triangle_select(self, event):
        self.clear_all_selections(except_widget=self.tri_list)
        sel = self.tri_list.curselection()
        if not sel:
            self.highlight_nodes = None
        else:
            idx = sel[0]
            self.highlight_nodes = set(self.triangles[idx])
        self.draw_graph()

    def on_path_select(self, event):
        self.clear_all_selections(except_widget=self.path_list)
        sel = self.path_list.curselection()
        if not sel:
            self.highlight_nodes = None
        else:
            idx = sel[0]
            # paths are (center, s1, s2)
            self.highlight_nodes = set(self.paths[idx])
        self.draw_graph()

    def on_g4_select(self, event, motif_key):
        self.clear_all_selections(except_widget=event.widget)
        sel = event.widget.curselection()
        if not sel:
            self.highlight_nodes = None
        else:
            idx = sel[0]
            quad = self.motifs4[motif_key][idx]
            self.highlight_nodes = set(quad)
        self.draw_graph()
    
    def on_close(self):
        plt.close('all')
        self.destroy()
        self.quit()


# ---------- main ----------
def main():
    # generate a random geometric graph
    n = 40          # number of nodes
    radius = 0.3    # connection radius
    seed = 42

    G = nx.random_geometric_graph(n, radius, seed=seed)
    pos = nx.get_node_attributes(G, "pos")

    # enumerate 3-node graphlets
    triangles, paths = enumerate_3node_graphlets(G)
    print(f"Found {len(triangles)} triangles and {len(paths)} 2-edge paths.")

    # enumerate 4-node graphlets
    motifs4 = enumerate_4node_graphlets(G)
    for k in sorted(motifs4.keys()):
        print(f"{k}: {len(motifs4[k])} motifs")

    app = GraphletViewer(G, pos, triangles, paths, motifs4)
    app.mainloop()


if __name__ == "__main__":
    main()
