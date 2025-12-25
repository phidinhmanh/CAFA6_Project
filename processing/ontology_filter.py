# processing/ontology_filter.py
from collections import defaultdict
from tqdm import tqdm
import gc
import obonet
import networkx as nx
import pandas as pd

from models.base import BaseStep
from data_paths import DataPaths


class OntologyFilter(BaseStep):
    step_name = "OntologyFilter"

    def __init__(self, min_score=0.001, top_k=1500, propagation_rounds=3):
        self.min_score = min_score
        self.top_k = top_k
        self.propagation_rounds = propagation_rounds

    # ===============================
    # CORE
    # ===============================
    def execute(self):
        parents = self._load_ontology()
        valid_tax = self._build_tax_rules()
        test_tax = self._load_test_tax()

        df = self.read_tsv(self.get_ctx("prediction_path"))

        rows = []
        for pid, group in tqdm(df.groupby("id")):
            preds = dict(zip(group.term, group.score))
            preds = self._filter_tax(pid, preds, test_tax, valid_tax)
            preds = self._propagate(preds, parents)
            preds = self._truncate(preds)
            rows.extend(self._format(pid, preds))

        out = self.get_cfg("final_output", "submission.tsv")
        self.write_tsv(rows, out)
        self.set_output_path(out)

        return self.context

    # ===============================
    # LOGIC BLOCKS
    # ===============================
    def _load_ontology(self):
        graph = obonet.read_obo(DataPaths.get("go-basic.obo"))
        return {n: set(graph.successors(n)) for n in graph.nodes()}

    def _build_tax_rules(self):
        tax = pd.read_csv(
            DataPaths.get("train_taxonomy"),
            sep="\t",
            names=["id", "tax"],
        )
        terms = pd.read_csv(
            DataPaths.get("train_terms"),
            sep="\t",
            usecols=["EntryID", "term"],
        )

        id2tax = dict(zip(tax.id, tax.tax))
        rules = defaultdict(set)

        for r in terms.itertuples():
            if r.EntryID in id2tax:
                rules[id2tax[r.EntryID]].add(r.term)

        del tax, terms
        gc.collect()
        return rules

    def _load_test_tax(self):
        df = pd.read_csv(
            DataPaths.get("testsuperset-taxon-list.tsv"),
            sep="\t",
            names=["id", "tax"],
        )
        return dict(zip(df.id, df.tax))

    def _filter_tax(self, pid, preds, test_tax, rules):
        tax = test_tax.get(pid)
        if tax in rules:
            return {t: s for t, s in preds.items() if t in rules[tax]}
        return preds

    def _propagate(self, preds, parents):
        for _ in range(self.propagation_rounds):
            updated = False
            for t, s in list(preds.items()):
                if s < self.min_score:
                    continue
                for p in parents.get(t, []):
                    if preds.get(p, 0) < s:
                        preds[p] = s
                        updated = True
            if not updated:
                break
        return preds

    def _truncate(self, preds):
        return dict(
            sorted(preds.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        )

    def _format(self, pid, preds):
        return [
            f"{pid}\t{t}\t{s:.3f}\n" for t, s in preds.items() if s > self.min_score
        ]
