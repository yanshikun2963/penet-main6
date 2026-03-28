#!/usr/bin/env python3
"""
DAPC (Drift-Aware Prototype Calibration) Patch Script
=====================================================
This script patches PE-NET's PrototypeEmbeddingNetwork to add:
  - DCM (Drift-Compensated Margin): drift-aware dynamic margin for loss_dis
  - DAT (Drift-Adaptive Temperature): drift-aware per-class temperature for cosine logits

Usage:
  python3 dapc_patch.py <path_to_penet_root> --alpha 0.1 --beta_t 0.1

  alpha=0, beta_t=0  => pure CB-Loss baseline (no DCM/DAT effect)
  alpha=0.1, beta_t=0.1 => ultra-conservative DCM+DAT (target: ±0.5 mR impact)
"""

import argparse
import os
import sys

def patch_predictors(filepath, alpha, beta_t):
    """Patch roi_relation_predictors.py to add DCM + DAT"""
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # ===================================================================
    # PATCH 1: Add drift computation in __init__ (after logit_scale, line 105)
    # ===================================================================
    old_init = """        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels"""
    
    new_init = f"""        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### DAPC: Drift-Aware Prototype Calibration parameters
        self.dapc_alpha = {alpha}   # DCM: margin modulation strength
        self.dapc_beta_t = {beta_t}  # DAT: temperature modulation strength
        self.proto_init = None  # will be computed on first forward pass
        self.drift_ema = None   # EMA-smoothed per-class drift signal
        self.drift_ema_momentum = 0.999
        #####

        ##### refine object labels"""
    
    if old_init not in content:
        print(f"ERROR: Cannot find __init__ anchor in {filepath}")
        return False
    content = content.replace(old_init, new_init)
    
    # ===================================================================
    # PATCH 2: Add drift computation + DAT in forward (before rel_dists line 203)
    #          Modify the cosine similarity line to include per-class temperature
    # ===================================================================
    old_forward_cosine = """        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        # the rel_dists will be used to calculate the Le_sim with the ce_loss"""
    
    new_forward_cosine = """        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### DAPC: Compute prototype drift from GloVe initialization
        with torch.no_grad():
            if self.proto_init is None:
                # First forward: compute initial prototypes from GloVe embeddings
                # Use W_pred to map GloVe -> prototype space (same as line 190)
                glove_proto = self.W_pred(self.rel_embed.weight.detach().clone())
                self.proto_init = glove_proto.detach().clone()
            
            # Current prototypes (before project_head, using W_pred output)
            current_proto = self.W_pred(self.rel_embed.weight)
            
            # Per-class cosine drift: how far each prototype moved from GloVe init
            cos_sim = F.cosine_similarity(current_proto, self.proto_init, dim=1)
            drift = (1.0 - cos_sim).clamp(min=0)  # [num_rel_cls]
            drift_max = drift.max().clamp(min=1e-8)
            normalized_drift = drift / drift_max  # [0, 1], head classes near 1, tail near 0
            
            # EMA smoothing for training stability
            if self.drift_ema is None:
                self.drift_ema = normalized_drift.clone()
            else:
                self.drift_ema = self.drift_ema_momentum * self.drift_ema + (1 - self.drift_ema_momentum) * normalized_drift

        ### DAT: Drift-Adaptive Temperature scaling
        # Head classes (high drift) get slightly suppressed logits
        # Tail classes (low drift) keep original scale
        # s_c = 1 / (1 + beta_t * drift_c), range: [1/(1+beta_t), 1.0]
        dat_scale = 1.0 / (1.0 + self.dapc_beta_t * self.drift_ema)  # [num_rel_cls]
        
        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        raw_logits = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ
        rel_dists = raw_logits * dat_scale.unsqueeze(0)  # per-class temperature modulation
        # the rel_dists will be used to calculate the Le_sim with the ce_loss"""
    
    if old_forward_cosine not in content:
        print(f"ERROR: Cannot find cosine similarity anchor in {filepath}")
        return False
    content = content.replace(old_forward_cosine, new_forward_cosine)
    
    # ===================================================================
    # PATCH 3: Add DCM in loss_dis (modify gamma1 from fixed 1.0 to drift-aware)
    # ===================================================================
    old_loss_dis = """            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), 51).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end """
    
    new_loss_dis = """            ###  Prototype-based Learning  ---- Euclidean distance
            ### DCM: Drift-Compensated Margin
            # Tail classes (low drift) get larger margin to enforce prototype separation
            # Head classes (high drift) keep original margin
            # gamma_c = 1.0 * (1 + alpha * (1 - drift_c/drift_max))
            rel_labels = cat(rel_labels, dim=0)
            with torch.no_grad():
                per_class_margin = 1.0 * (1.0 + self.dapc_alpha * (1.0 - self.drift_ema))  # [num_rel_cls]
                # Per-sample margin based on its GT label
                gamma1_per_sample = per_class_margin[rel_labels]  # [batch_size]
            
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 51, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), 51).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1_per_sample).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma_c)
            ### end """
    
    if old_loss_dis not in content:
        print(f"ERROR: Cannot find loss_dis anchor in {filepath}")
        return False
    content = content.replace(old_loss_dis, new_loss_dis)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"OK: Patched {filepath}")
    print(f"    alpha={alpha} (DCM margin strength)")
    print(f"    beta_t={beta_t} (DAT temperature strength)")
    return True


def verify_patch(filepath):
    """Verify the patch was applied correctly"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    checks = [
        ("dapc_alpha", "DAPC alpha parameter"),
        ("dapc_beta_t", "DAPC beta_t parameter"),
        ("proto_init", "proto_init initialization"),
        ("drift_ema", "drift EMA"),
        ("dat_scale", "DAT temperature scaling"),
        ("per_class_margin", "DCM per-class margin"),
        ("gamma1_per_sample", "DCM per-sample margin"),
        ("raw_logits", "raw logits before DAT"),
    ]
    
    all_ok = True
    for keyword, description in checks:
        if keyword not in content:
            print(f"  FAIL: Missing {description} ({keyword})")
            all_ok = False
        else:
            print(f"  OK: {description}")
    
    # Verify CB-Loss is untouched
    if "cb_loss_beta=0.9999" in open(filepath.replace("roi_relation_predictors.py", "loss.py")).read():
        print(f"  OK: CB-Loss beta=0.9999 unchanged in loss.py")
    else:
        print(f"  WARN: CB-Loss beta not found in loss.py")
    
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAPC Patch for PE-NET")
    parser.add_argument("penet_root", help="Path to PE-NET root directory")
    parser.add_argument("--alpha", type=float, default=0.1, help="DCM margin strength (default: 0.1)")
    parser.add_argument("--beta_t", type=float, default=0.1, help="DAT temperature strength (default: 0.1)")
    args = parser.parse_args()
    
    predictor_path = os.path.join(
        args.penet_root,
        "maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py"
    )
    
    if not os.path.exists(predictor_path):
        print(f"ERROR: File not found: {predictor_path}")
        sys.exit(1)
    
    print(f"Patching {predictor_path}")
    print(f"Parameters: alpha={args.alpha}, beta_t={args.beta_t}")
    print("-" * 60)
    
    success = patch_predictors(predictor_path, args.alpha, args.beta_t)
    if success:
        print("-" * 60)
        print("Verifying patch...")
        verify_patch(predictor_path)
        print("-" * 60)
        print("DONE. CB-Loss (beta=0.9999) is untouched in loss.py.")
        print("DCM and DAT are added with ultra-conservative parameters.")
    else:
        print("PATCH FAILED. Please check error messages above.")
        sys.exit(1)
