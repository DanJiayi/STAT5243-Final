import torch
import torch.nn as nn
import torch.nn.functional as F

# encoder / decoder / gating / nb reconstruction
n_species = 2   # frog / zebrafish

def glorot_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
class SmallGate(nn.Module):
    """
    Species-specific gating module.
    For each species k, learns per-dimension gate g_k in [0, 2].
    """
    def __init__(self, dim, min_gate=0.0005, max_gate=1.9995):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.zeros(dim))
        self.min_gate = min_gate
        self.max_gate = max_gate

    def forward(self, h):
        gate = torch.sigmoid(self.logit_scale) * 2.0
        gate = torch.clamp(gate, self.min_gate, self.max_gate)
        return h * gate


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.apply(glorot_init)

    def forward(self, x):
        return self.net(x)  # (B, H)


class XSVAE(nn.Module):
    """
    Clean XSVAE (correct version):
    - No concat species_emb
    - Species-specific gate optional
    """
    def __init__(
        self,
        input_dim,
        hidden_dims,
        latent_dim,
        n_species,
        n_labels,
        recon_mode="nb",
        use_species_gate=True,
        use_species_pred=False
    ):
        super().__init__()
        self.recon_mode = recon_mode
        self.n_species = n_species
        self.use_species_gate = use_species_gate
        self.use_species_pred = use_species_pred

        # Encoder
        self.encoder = Encoder(input_dim, hidden_dims)

        # Species-specific gate
        if use_species_gate:
            self.spec_gate = nn.ModuleList([
                SmallGate(hidden_dims[-1]) for _ in range(n_species)
            ])
        else:
            self.spec_gate = None

        # Latent projection (no species_emb involved)
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder (input = z only)
        dims = [latent_dim] + hidden_dims[::-1]
        dec_layers = []
        for i in range(len(dims)-1):
            dec_layers.append(nn.Linear(dims[i], dims[i+1]))
            dec_layers.append(nn.ReLU())
        dec_layers.append(nn.Linear(dims[-1], input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # NB dispersion
        self.px_r = nn.Parameter(torch.ones(input_dim) * 5.0)

        # Classifier on z only
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_labels)
        )

        # Optional species predictor (only on gated h)
        if use_species_pred:
            self.spec_predictor = nn.Sequential(
                nn.Linear(hidden_dims[-1], 32),
                nn.ReLU(),
                nn.Linear(32, n_species)
            )
        else:
            self.spec_predictor = None

        self.apply(glorot_init)


    def encode(self, x, s):
        h = self.encoder(x)  # (B, H)
        h2 = h.clone()

        # Apply species gate
        if self.use_species_gate:
            for k in range(self.n_species):
                mask = (s == k)
                if mask.any():
                    h2[mask] = self.spec_gate[k](h[mask])

        mu = self.mu(h2)
        logvar = self.logvar(h2)
        return mu, logvar, h2


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z):
        out = self.decoder(z)
        if self.recon_mode == "nb":
            return F.softplus(out) + 1e-4
        else:
            return out


    def forward(self, x, s):
        mu, logvar, h2 = self.encode(x, s)
        z = self.reparameterize(mu, logvar)
        mean = self.decode(z)

        logits = self.classifier(z)

        if self.use_species_pred:
            species_logits = self.spec_predictor(h2)
        else:
            species_logits = None

        return {
            "mean": mean,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "logits": logits,
            "species_logits": species_logits
        }
