# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional, List, Dict

import tkinter as tk
from tkinter import ttk, messagebox

from data_loader import load_movies
from recommender import ContentBasedRecommender


class RecommenderApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Netflix Content-Based Recommender")
        self.geometry("1040x800")
        self.minsize(980, 740)

        self.movies_df = None
        self.model: Optional[ContentBasedRecommender] = None

        self.all_titles: List[str] = []
        self.titles: List[str] = []
        self.all_genres: List[str] = []

        self.topn_var = tk.StringVar(value="25")
        self.search_var = tk.StringVar()
        self.genre_var = tk.StringVar(value="All genres")

        self.last_selected_titles: List[str] = []
        self.explanations: Dict[str, str] = {}

        # UI state
        self.status_var = tk.StringVar(value="Ready.")
        self._last_rec_count = 0

        self._apply_theme()

        self._load_data_and_model()
        self._build_ui()

        # Keyboard shortcuts (UX only)
        self.bind_all("<Control-f>", self._focus_search)
        self.bind_all("<Return>", self._recommend_shortcut)
        self.bind_all("<Escape>", self._clear_search_shortcut)

    def _apply_theme(self):
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.option_add("*Font", ("Segoe UI", 10))

        # Palette
        self.COL_BG = "#F5F6FA"
        self.COL_CARD = "#FFFFFF"
        self.COL_TEXT = "#111827"
        self.COL_MUTED = "#6B7280"
        self.COL_ACCENT = "#E50914"     # Netflix red
        self.COL_ACCENT_D = "#B20710"
        self.COL_BORDER = "#E5E7EB"
        self.COL_SELECT_BG = "#DBEAFE"
        self.COL_SELECT_FG = "#111827"

        self.configure(bg=self.COL_BG)

        style.configure("TLabelframe", padding=(12, 12), background=self.COL_BG)
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), background=self.COL_BG)

        # Frames / labels
        style.configure("TFrame", background=self.COL_BG)
        style.configure("TLabel", background=self.COL_BG, foreground=self.COL_TEXT)

        style.configure("TEntry", padding=(10, 7))
        style.configure("TCombobox", padding=(10, 6))

        style.configure("TButton", padding=(12, 8))

        style.configure(
            "Accent.TButton",
            foreground="white",
            background=self.COL_ACCENT,
            borderwidth=0,
            focusthickness=0
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.COL_ACCENT_D), ("pressed", "#8A050C")],
            foreground=[("disabled", "#e5e7eb")]
        )

        style.configure(
            "Secondary.TButton",
            foreground=self.COL_TEXT,
            background="#E5E7EB",
            borderwidth=0,
            focusthickness=0
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#D1D5DB"), ("pressed", "#9CA3AF")]
        )

        style.configure("Treeview", rowheight=28, fieldbackground=self.COL_CARD, background=self.COL_CARD)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.map(
            "Treeview",
            background=[("selected", self.COL_SELECT_BG)],
            foreground=[("selected", self.COL_SELECT_FG)],
        )

        style.configure("TSeparator", background=self.COL_BORDER)

    def _load_data_and_model(self):
        project_root = Path(__file__).parent
        csv_path = project_root / "data" / "movies.csv"

        try:
            self.movies_df = load_movies(str(csv_path))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load movies data:\n{e}")
            self.destroy()
            return

        try:
            self.model = ContentBasedRecommender()
            self.model.fit(self.movies_df)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize recommender:\n{e}")
            self.destroy()
            return

        self.all_titles = sorted(self.movies_df["title"].tolist())
        self.titles = self.all_titles.copy()

        genre_tokens = set()
        if "genres" in self.movies_df.columns:
            for g in self.movies_df["genres"]:
                for token in str(g).split():
                    token = token.strip()
                    if token:
                        genre_tokens.add(token)

        self.all_genres = ["All genres"] + sorted(genre_tokens)
        self.genre_var.set("All genres")

    def _build_ui(self):
        # Root container
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        header = tk.Frame(root, bg=self.COL_CARD, highlightbackground=self.COL_BORDER, highlightthickness=1)
        header.pack(fill="x", pady=(0, 12))

        # Left red accent stripe
        stripe = tk.Frame(header, bg=self.COL_ACCENT, width=6)
        stripe.pack(side="left", fill="y")

        header_inner = tk.Frame(header, bg=self.COL_CARD)
        header_inner.pack(side="left", fill="both", expand=True, padx=12, pady=10)

        title_lbl = tk.Label(
            header_inner,
            text="Netflix Content-Based Recommender",
            bg=self.COL_CARD,
            fg=self.COL_TEXT,
            font=("Segoe UI", 14, "bold")
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            header_inner,
            text="Pick one or more titles, then get recommendations with explanations.",
            bg=self.COL_CARD,
            fg=self.COL_MUTED,
            font=("Segoe UI", 10)
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        self.paned = ttk.PanedWindow(root, orient="vertical")
        self.paned.pack(fill="both", expand=True)

        top_card = ttk.Frame(self.paned)
        bottom_card = ttk.Frame(self.paned)

        self.paned.add(top_card, weight=3)
        self.paned.add(bottom_card, weight=4)

        top_frame = ttk.LabelFrame(top_card, text="Pick movies / series you like")
        top_frame.pack(fill="both", expand=True)

        filters_frame = ttk.Frame(top_frame)
        filters_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(filters_frame, text="Search title:").pack(side="left")
        self.search_entry = ttk.Entry(filters_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side="left", padx=(8, 18))

        ttk.Label(filters_frame, text="Genre:").pack(side="left")
        genre_combo = ttk.Combobox(
            filters_frame,
            textvariable=self.genre_var,
            values=self.all_genres,
            state="readonly",
            width=20,
        )
        genre_combo.pack(side="left", padx=(8, 0))

        # Live filtering
        self.search_var.trace_add("write", self.on_filters_changed)
        genre_combo.bind("<<ComboboxSelected>>", self.on_filters_changed)

        catalog_frame = ttk.Frame(top_frame)
        catalog_frame.pack(fill="both", expand=True)

        self.catalog_tree = ttk.Treeview(
            catalog_frame,
            columns=("title", "desc"),
            show="headings",
            selectmode="extended",
        )
        self.catalog_tree.heading("title", text="Title")
        self.catalog_tree.heading("desc", text="Description")
        self.catalog_tree.column("title", width=280, anchor="w", stretch=False)
        self.catalog_tree.column("desc", width=680, anchor="w", stretch=True)

        cat_scroll = ttk.Scrollbar(catalog_frame, orient="vertical", command=self.catalog_tree.yview)
        self.catalog_tree.configure(yscrollcommand=cat_scroll.set)

        self.catalog_tree.pack(side="left", fill="both", expand=True)
        cat_scroll.pack(side="right", fill="y")

        self._update_catalog()
        self._apply_row_stripes(self.catalog_tree)

        self.catalog_tree.bind("<Double-1>", lambda e: self.on_recommend())

        bottom_header = ttk.Frame(bottom_card)
        bottom_header.pack(fill="x", pady=(0, 10))

        ttk.Label(bottom_header, text="Number of recommendations:").pack(side="left")

        topn_combo = ttk.Combobox(
            bottom_header,
            textvariable=self.topn_var,
            values=["3", "5", "10", "25", "50", "100"],
            width=6,
            state="readonly",
        )
        topn_combo.pack(side="left", padx=(8, 14))

        recommend_btn = ttk.Button(
            bottom_header, text="Recommend", command=self.on_recommend, style="Accent.TButton"
        )
        recommend_btn.pack(side="left")

        clear_btn = ttk.Button(
            bottom_header, text="Clear results", command=self.clear_results, style="Secondary.TButton"
        )
        clear_btn.pack(side="left", padx=(10, 0))

        split = ttk.PanedWindow(bottom_card, orient="horizontal")
        split.pack(fill="both", expand=True)

        left = ttk.Frame(split)
        right = ttk.Frame(split)
        split.add(left, weight=4)
        split.add(right, weight=2)

        rec_frame = ttk.LabelFrame(left, text="Recommendations")
        rec_frame.pack(fill="both", expand=True)

        columns = ("rank", "title", "score", "bar")
        self.tree = ttk.Treeview(rec_frame, columns=columns, show="headings", height=14)
        self.tree.heading("rank", text="№")
        self.tree.heading("title", text="Title")
        self.tree.heading("score", text="Similarity")
        self.tree.heading("bar", text="Match")

        self.tree.column("rank", width=45, anchor="center", stretch=False)
        self.tree.column("title", width=380, anchor="w", stretch=True)
        self.tree.column("score", width=90, anchor="center", stretch=False)
        self.tree.column("bar", width=220, anchor="w", stretch=True)

        self.tree.pack(side="left", fill="both", expand=True)

        tree_scrollbar = ttk.Scrollbar(rec_frame, orient="vertical", command=self.tree.yview)
        tree_scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        # Similarity color tags
        self.tree.tag_configure("top1", background="#111827", foreground="white")
        self.tree.tag_configure("s_high", background="#FEE2E2")   # light red
        self.tree.tag_configure("s_mid", background="#FEF3C7")    # light amber
        self.tree.tag_configure("s_low", background="#DCFCE7")    # light green
        self.tree.tag_configure("s_zero", background="#F3F4F6")   # light gray

        self.tree.bind("<<TreeviewSelect>>", self.on_select_recommendation)

        details_outer = tk.Frame(right, bg=self.COL_BG)
        details_outer.pack(fill="both", expand=True)

        details_card = tk.Frame(details_outer, bg=self.COL_CARD, highlightbackground=self.COL_BORDER, highlightthickness=1)
        details_card.pack(fill="both", expand=True)

        details_header = tk.Frame(details_card, bg=self.COL_CARD)
        details_header.pack(fill="x", padx=12, pady=(10, 6))

        self.details_title = tk.Label(
            details_header,
            text="Details",
            bg=self.COL_CARD,
            fg=self.COL_TEXT,
            font=("Segoe UI", 11, "bold")
        )
        self.details_title.pack(anchor="w")

        self.details_meta = tk.Label(
            details_header,
            text="Select a recommendation to see description + explanation.",
            bg=self.COL_CARD,
            fg=self.COL_MUTED,
            font=("Segoe UI", 9)
        )
        self.details_meta.pack(anchor="w", pady=(2, 0))

        ttk.Separator(details_card, orient="horizontal").pack(fill="x", padx=12, pady=(0, 8))

        # Scrollable text
        text_wrap = tk.Frame(details_card, bg=self.COL_CARD)
        text_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.details_text = tk.Text(
            text_wrap,
            wrap="word",
            height=10,
            borderwidth=0,
            highlightthickness=0,
            bg=self.COL_CARD,
            fg=self.COL_TEXT,
            font=("Segoe UI", 10)
        )
        self.details_text.pack(side="left", fill="both", expand=True)

        details_scroll = ttk.Scrollbar(text_wrap, orient="vertical", command=self.details_text.yview)
        details_scroll.pack(side="right", fill="y")
        self.details_text.configure(yscrollcommand=details_scroll.set)

        self._set_details_text("Pick titles above, click Recommend.\n\nThen click a result to see details here.")
        self.details_text.configure(state="disabled")

        # ---------- Status bar ----------
        status = tk.Frame(root, bg=self.COL_BG)
        status.pack(fill="x", pady=(10, 0))

        self.status_label = tk.Label(
            status, textvariable=self.status_var, bg=self.COL_BG, fg=self.COL_MUTED, anchor="w"
        )
        self.status_label.pack(fill="x")

        self._update_status()

    def _apply_row_stripes(self, tree: ttk.Treeview):
        tree.tag_configure("odd", background="#ffffff")
        tree.tag_configure("even", background="#F6F7FB")
        for i, iid in enumerate(tree.get_children()):
            tree.item(iid, tags=("even" if i % 2 == 0 else "odd",))

    def clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.explanations.clear()
        self._last_rec_count = 0
        self._set_details_header("Details", "Select a recommendation to see details.")
        self._set_details_text("No results. Pick titles above and click Recommend.")
        self._update_status()

    def _score_to_bar(self, score: float) -> str:
        score = max(0.0, min(1.0, float(score)))
        length = 18
        filled = int(round(score * length))
        return "█" * filled + "░" * (length - filled)

    def _get_genres_for_title(self, title: str) -> List[str]:
        if self.movies_df is None:
            return []
        rows = self.movies_df[self.movies_df["title"] == title]
        if rows.empty:
            return []
        genres_str = str(rows.iloc[0]["genres"])
        return [g.strip() for g in genres_str.split() if g.strip()]

    def _get_description_for_title(self, title: str) -> str:
        if self.movies_df is None:
            return ""
        rows = self.movies_df[self.movies_df["title"] == title]
        if rows.empty:
            return ""
        return str(rows.iloc[0].get("description", "")).strip()

    def _update_catalog(self):
        for item in self.catalog_tree.get_children():
            self.catalog_tree.delete(item)

        for title in self.titles:
            desc = self._get_description_for_title(title) or ""
            desc_one_line = " ".join(desc.split())
            if len(desc_one_line) > 140:
                desc_one_line = desc_one_line[:137] + "..."
            self.catalog_tree.insert("", "end", values=(title, desc_one_line))

        self._apply_row_stripes(self.catalog_tree)
        self._update_status()

    def _set_details_header(self, title: str, meta: str):
        self.details_title.config(text=title)
        self.details_meta.config(text=meta)

    def _set_details_text(self, text: str):
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", text)
        self.details_text.configure(state="disabled")

    def _update_status(self):
        total = len(self.all_titles)
        shown = len(self.titles)

        sel = len(self.catalog_tree.selection()) if hasattr(self, "catalog_tree") else 0
        recs = self._last_rec_count

        self.status_var.set(
            f"Titles: {shown}/{total} shown • Selected: {sel} • Recommendations: {recs} "
            f"• Shortcuts: Ctrl+F search, Enter recommend, Esc clear search"
        )

    # -------------------- events --------------------
    def on_filters_changed(self, *args):
        query = self.search_var.get().lower().strip()
        selected_genre = self.genre_var.get()

        filtered: List[str] = []
        for title in self.all_titles:
            if query and query not in title.lower():
                continue

            if selected_genre and selected_genre != "All genres":
                genres = self._get_genres_for_title(title)
                if selected_genre not in genres:
                    continue

            filtered.append(title)

        self.titles = filtered
        self._update_catalog()

    def on_recommend(self):
        if self.model is None:
            messagebox.showerror("Error", "Model is not initialized.")
            return

        selected_items = self.catalog_tree.selection()
        if not selected_items:
            messagebox.showwarning("Missing input", "Please select at least one title.")
            return

        selected_titles = [self.catalog_tree.item(i, "values")[0] for i in selected_items]
        self.last_selected_titles = selected_titles

        try:
            top_n = int(self.topn_var.get())
        except ValueError:
            messagebox.showwarning("Invalid input", "Top N must be an integer.")
            return

        try:
            if hasattr(self.model, "recommend_by_titles"):
                recs = self.model.recommend_by_titles(selected_titles, top_n=top_n)
            else:
                recs = self.model.recommend_by_title(selected_titles[0], top_n=top_n)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        user_genres = set()
        for t in selected_titles:
            user_genres.update(self._get_genres_for_title(t))

        # Clear table
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.explanations.clear()

        # Insert results (same logic, only visuals)
        count = 0
        for idx, (rec_title, score) in enumerate(recs, start=1):
            bar = self._score_to_bar(score)

            rec_genres = set(self._get_genres_for_title(rec_title))
            common_genres = [g for g in rec_genres if g in user_genres]

            if common_genres:
                explanation = (
                    "Recommended because it shares common genres with your selected titles: "
                    + ", ".join(common_genres) + "."
                )
            else:
                explanation = "Recommended because its description is similar to your selected titles."

            if idx == 1:
                tag = "top1"
            else:
                if score >= 0.20:
                    tag = "s_high"
                elif score >= 0.08:
                    tag = "s_mid"
                elif score > 0.0:
                    tag = "s_low"
                else:
                    tag = "s_zero"

            item_id = self.tree.insert("", "end", values=(idx, rec_title, f"{score:.3f}", bar), tags=(tag,))
            self.explanations[item_id] = explanation
            count += 1

        self._last_rec_count = count
        self._set_details_header("Details", "Click a result to see description + explanation.")
        self._set_details_text("Recommendations generated. Select one item from the table.")
        self._update_status()

    def on_select_recommendation(self, event):
        selection = self.tree.selection()
        if not selection:
            return

        item_id = selection[0]
        values = self.tree.item(item_id, "values")
        rec_title = values[1] if len(values) >= 4 else values[0]
        score_str = values[2] if len(values) >= 3 else ""

        desc = self._get_description_for_title(rec_title) or "No description available."
        explanation = self.explanations.get(item_id, "No explanation available for this recommendation.")

        self._set_details_header(rec_title, f"Similarity: {score_str}")
        self._set_details_text(f"{desc}\n\n{explanation}")

    def _focus_search(self, event=None):
        if hasattr(self, "search_entry"):
            self.search_entry.focus_set()
            self.search_entry.select_range(0, "end")
        return "break"

    def _recommend_shortcut(self, event=None):
        w = self.focus_get()
        if isinstance(w, tk.Text):
            return
        self.on_recommend()
        return "break"

    def _clear_search_shortcut(self, event=None):
        # Clear search only (genre stays)
        self.search_var.set("")
        return "break"


if __name__ == "__main__":
    app = RecommenderApp()
    app.mainloop()
