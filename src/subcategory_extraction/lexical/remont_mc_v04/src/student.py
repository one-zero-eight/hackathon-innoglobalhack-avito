"""Ученик на CPU: ONNX-энкодер (мульти-лейбл 102–106), без torch/transformers — только onnxruntime + tokenizers.
Модель: models/student/<tag>/{model_int8.onnx | model.onnx, tokenizer.json, meta.json} (tools/train_student.py)."""
from __future__ import annotations
import json, os, pathlib
import numpy as np

CATS = [102, 103, 104, 105, 106]


class Student:
    def __init__(self, path, int8: bool = True, threads: int | None = None, device: str = 'cpu'):
        """device='gpu' — только для оценки: модель (fp32) считается на GPU Modal (tools/gpu_infer.py)."""
        p = pathlib.Path(path)
        self.meta = json.loads((p / 'meta.json').read_text())
        self.max_len = self.meta['max_len']           # как в обучении (включая [CLS]/[SEP])
        self.inputs = self.meta['inputs']
        self.device = device
        if device == 'gpu':
            import sys, modal
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
            import gpu_infer
            if not gpu_infer.uploaded(p.name): gpu_infer.upload(p)
            self.tag = p.name
            self._remote = modal.Cls.from_name('remont-infer', 'Infer')(tag=p.name).proba
            return
        import onnxruntime as ort
        from tokenizers import Tokenizer
        f = p / ('model_int8.onnx' if int8 and (p / 'model_int8.onnx').exists() else 'model.onnx')
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or min(8, os.cpu_count() or 1)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(f), so, providers=['CPUExecutionProvider'])
        self.tok = Tokenizer.from_file(str(p / 'tokenizer.json'))
        self.tok.enable_truncation(self.max_len)
        pad_id = self.tok.token_to_id('[PAD]')
        if pad_id is None: pad_id = self.tok.token_to_id('<pad>') or 0
        self.pad_id = pad_id

    def proba(self, texts, batch: int = 8) -> np.ndarray:
        """texts — очищенные тексты (mc_pipeline.clean). -> массив [n, 5] вероятностей по CATS.
        Режим 'cat' (пары): по одному проходу на каждую категорию с подсветкой упоминаний (mc_pipeline.mark_doc)."""
        if __package__:
            from . import mc_pipeline as P
        else:
            import mc_pipeline as P
        if self.meta.get('rules_prefix'):
            texts = [P.rules_prefix(t or '') + (t or '') for t in texts]
        if self.meta.get('mode') == 'cat':
            pairs = [P.mark_doc(t or '', c) for t in texts for c in CATS]
            return self._run(pairs, batch * 2).reshape(len(texts), len(CATS))
        return self._run(texts, batch)

    def _run(self, texts, batch):
        """Сырые входы -> вероятности [n, число выходов]. Сортировка по длине — меньше паддинга."""
        texts = [t or '' for t in texts]
        n_out = len(self.meta['cats'])
        if self.device == 'gpu':
            return np.array(self._remote.remote(texts, self.max_len, self.inputs, n_out), dtype=np.float32)
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), n_out), dtype=np.float32)
        for k in range(0, len(order), batch):
            ids = order[k:k + batch]
            enc = self.tok.encode_batch([texts[i] for i in ids])
            L = max(len(e.ids) for e in enc)
            feed = {'input_ids': np.full((len(ids), L), self.pad_id, np.int64),
                    'attention_mask': np.zeros((len(ids), L), np.int64)}
            if 'token_type_ids' in self.inputs:
                feed['token_type_ids'] = np.zeros((len(ids), L), np.int64)
            for j, e in enumerate(enc):
                n = len(e.ids)
                feed['input_ids'][j, :n] = e.ids; feed['attention_mask'][j, :n] = 1
            logits = self.sess.run(['logits'], {k: v for k, v in feed.items() if k in self.inputs})[0]
            out[ids] = 1 / (1 + np.exp(-logits))
        return out
