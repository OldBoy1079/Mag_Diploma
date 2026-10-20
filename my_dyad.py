import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score


# Датасет
class BatteryDataset(Dataset):
    """
    Читает .pkl сниппеты.
    """
    def __init__(self, all_car_dict_path, ind_ood_path,
                 train=True, fold_num=0, norm_stats=None):
        d = np.load(ind_ood_path, allow_pickle=True).item()
        ind, ood = d['ind_sorted'], d['ood_sorted']
        lo, hi = int(fold_num * len(ind) / 5), int((fold_num + 1) * len(ind) / 5)
        car_ids = ind[:lo] + ind[hi:] if train else ind[lo:hi] + ood

        all_cars = np.load(all_car_dict_path, allow_pickle=True).item()
        self.items = []
        for c in car_ids:
            for p in all_cars[c]:
                self.items.append(torch.load(p, weights_only=False))

        if norm_stats is not None:
            self.mean, self.std = norm_stats
        elif train:
            k = min(200, len(self.items))
            sample = np.stack([np.asarray(self.items[i][0], dtype=np.float32) for i in range(k)])
            self.mean = sample.mean(axis=(0, 1))
            self.std  = sample.std(axis=(0, 1)) + 1e-4
        else:
            self.mean, self.std = 0.0, 1.0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, meta = self.items[i]
        x = np.asarray(x, dtype=np.float32)
        x = (x - self.mean) / self.std
        meta = dict(meta)
        meta['label_bin'] = 0 if meta['label'] == '00' else 1
        return torch.from_numpy(x), meta


# Модель
class DyAD(nn.Module):
    def __init__(self, enc_idx, dec_idx, tgt_idx,
                 hidden=32, latent=32, num_layers=3, rnn_type='gru'):
        super().__init__()
        self.enc_idx, self.dec_idx, self.tgt_idx = enc_idx, dec_idx, tgt_idx
        rnn = {'gru': nn.GRU, 'lstm': nn.LSTM, 'rnn': nn.RNN}[rnn_type]

        d_in  = len(enc_idx)   # сколько признаков видит энкодер
        d_dec = len(dec_idx)   # сколько признаков на входе декодера
        d_out = len(tgt_idx)   # сколько признаков восстанавливаем

        self.encoder = rnn(d_in, hidden, num_layers, batch_first=True)
        self.fc_mu     = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)

        self.fc_h0   = nn.Linear(latent, hidden * num_layers)
        self.decoder = rnn(d_dec, hidden, num_layers, batch_first=True)
        self.fc_out  = nn.Linear(hidden, d_out)

        self.mileage_head = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        self.num_layers, self.hidden, self.latent = num_layers, hidden, latent

    def _select(self, x, idx):
        return x[:, :, idx]

    def encode(self, x_full):
        _, h = self.encoder(x_full)
        h = h[-1]
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z, x0):
        h0 = torch.tanh(self.fc_h0(z)).view(-1, self.num_layers, self.hidden)
        h0 = h0.permute(1, 0, 2).contiguous()
        out, _ = self.decoder(x0, h0)
        return self.fc_out(out)

    def forward(self, x):
        x_full = self._select(x, self.enc_idx)   # что видит энкодер
        x0 = self._select(x, self.dec_idx)   # вход декодера
        x1 = self._select(x, self.tgt_idx)   # что восстанавливаем

        mu, logvar = self.encode(x_full)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        x1_hat = self.decode(z, x0)
        return x1_hat, x1, mu, logvar, z


def dyad_loss(x1_hat, x1, mu, logvar, z, mileage, mileage_pred,
              w_kl=0.05, w_mileage=0.05):
    rec = F.mse_loss(x1_hat, x1)
    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    mil = F.mse_loss(mileage_pred, mileage)
    return rec + w_kl * kl + w_mileage * mil, dict(rec=rec.item(), kl=kl.item(), mil=mil.item())


# Обучение
def train(model, loader, epochs=30, lr=1e-3, device='cuda'):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for x, meta in loader:
            x = x.to(device)
            mileage = torch.tensor(meta['mileage'], dtype=torch.float32, device=device)
            mileage = (mileage - mileage.min()) / (mileage.max() - mileage.min() + 1e-8)

            x1_hat, x1, mu, logvar, z = model(x)
            mileage_pred = model.mileage_head(z).squeeze(-1)
            loss, parts = dyad_loss(x1_hat, x1, mu, logvar, z, mileage, mileage_pred)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tot += loss.item(); n += 1
        sched.step()
        print(f'[Epoch {ep:03d}] loss={tot/n:.5f}')


# Извлечение ошибок + robust scoring
@torch.no_grad()
def extract_errors(model, loader, device='cuda'):
    model.to(device).eval()
    errs, cars, labels = [], [], []
    for x, meta in loader:
        x = x.to(device)
        x1_hat, x1, _, _, _ = model(x)
        e = ((x1_hat - x1) ** 2).mean(dim=(1, 2)).cpu().numpy()
        errs.append(e)
        cars.extend(meta['car'])
        labels.extend(meta['label_bin'])
    return (np.concatenate(errs),
            np.asarray(cars),
            np.asarray(labels, dtype=int))


def vehicle_scores(errors, cars, top_p=10.0):
    scores = {}
    for c in np.unique(cars):
        e = errors[cars == c]
        k = max(1, int(len(e) * top_p / 100))
        scores[int(c)] = float(np.sort(e)[-k:].mean())
    return scores


def evaluate_auroc(veh_scores, veh_labels):
    y = np.array([veh_labels[c] for c in veh_scores])
    s = np.array(list(veh_scores.values()))
    if y.min() == y.max():
        return float('nan')
    return roc_auc_score(y, s)



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--all_car_dict', default='../five_fold_utils/all_car_dict.npz.npy')
    p.add_argument('--ind_ood_dict', default='../five_fold_utils/ind_odd_dict3.npz.npy')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--enc_idx', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5])
    p.add_argument('--tgt_idx', type=int, nargs='+', default=[2, 3, 4, 5])
    args = p.parse_args()

    # данные
    train_ds = BatteryDataset(args.all_car_dict, args.ind_ood_dict,
                              train=True, fold_num=args.fold)
    norm_stats = (train_ds.mean, train_ds.std)
    test_ds = BatteryDataset(args.all_car_dict, args.ind_ood_dict,
                             train=False, fold_num=args.fold, norm_stats=norm_stats)

    def _collate(batch):
        xs = torch.stack([b[0] for b in batch], dim=0)
        metas = {
            'car':    [b[1]['car']          for b in batch],
            'label_bin': [b[1]['label_bin'] for b in batch],
            'mileage':   [b[1]['mileage']   for b in batch],
        }
        return xs, metas

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=_collate)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=_collate)

    # модель
    model = DyAD(enc_idx=args.enc_idx, dec_idx=args.enc_idx, tgt_idx=args.tgt_idx)
    print('params:', sum(p.numel() for p in model.parameters()))

    x_dbg, meta_dbg = next(iter(train_loader))
    print('x batch shape:', x_dbg.shape)
    print('dtype:', x_dbg.dtype)
    print('meta car:', meta_dbg['car'][:3])
    print('meta label_bin:', meta_dbg['label_bin'][:3])
    train(model, train_loader, epochs=args.epochs)

    # оценка
    err_te, car_te, lab_te = extract_errors(model, test_loader)
    car2label = {int(c): int(l) for c, l in zip(car_te, lab_te)}
    scores = vehicle_scores(err_te, car_te, top_p=10.0)
    print('AUROC (vehicle-level):', evaluate_auroc(scores, car2label))