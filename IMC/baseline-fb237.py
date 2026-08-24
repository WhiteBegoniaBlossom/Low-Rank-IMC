"""
Baseline models for IMC project on tyler-main's pre-split fb237 dataset versions.

Uses the same data loading and PLM embeddings as main-fb237.py, but instead of the
IMC matrix factorization approach, it constructs features by concatenating or
subtracting head/tail entity embeddings and trains standard classifiers
(Logistic Regression, Random Forest, LightGBM), plus a Feature Translation method.

Usage:
    python baseline-fb237.py --version fb237_v1
    python baseline-fb237.py --all
"""
import os
import sys
import pickle
import argparse
import time

# Limit OpenMP threads before lightgbm imports — on many-core servers
# (64+ CPUs) the default "use all cores" causes thread oversubscription
# and hangs during training.
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = str(min(os.cpu_count() or 1, 16))

import numpy as np
import cupy as cp
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from scipy.special import softmax

from SparseRelationMatrix import create_sparse_relation_matrices

# ============================================================
# Data loading (same as main-fb237.py)
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
EMBED_DIR = os.path.join(os.path.dirname(__file__), "embeddings")

V1_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
V2_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
V3_SEEDS = [7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010]
VERSIONS = [
    f"fb237_v1_ind_seed{s}" for s in V1_SEEDS
] + [
    f"fb237_v2_ind_seed{s}" for s in V2_SEEDS
] + [
    f"fb237_v3_ind_seed{s}" for s in V3_SEEDS
]


def load_fb237_data(version, model="roberta", aggregation="sum"):
    import pandas as pd

    version_dir = os.path.join(DATA_DIR, version)
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(f"Dataset directory not found: {version_dir}")

    train_triples = pd.read_csv(
        os.path.join(version_dir, "train.txt"), sep="\t",
        header=None, names=["head", "relation", "tail"]
    )
    valid_triples = pd.read_csv(
        os.path.join(version_dir, "valid.txt"), sep="\t",
        header=None, names=["head", "relation", "tail"]
    )
    test_triples = pd.read_csv(
        os.path.join(version_dir, "test.txt"), sep="\t",
        header=None, names=["head", "relation", "tail"]
    )

    emb_path = os.path.join(EMBED_DIR, f"{version}_{model}_{aggregation}_embeddings.pkl")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embeddings not found: {emb_path}. "
            f"Run generate_plm_embeddings.py --version {version} --model {model} --aggregation {aggregation} first."
        )
    with open(emb_path, "rb") as f:
        node2emb = pickle.load(f)

    print(f"[{version}] Train: {len(train_triples)}, Valid: {len(valid_triples)}, "
          f"Test: {len(test_triples)}, Embeddings: {len(node2emb)}")
    return train_triples, valid_triples, test_triples, node2emb


def prepare_features_fb237(train_triples, valid_triples, test_triples, node2emb, random_seed=28):
    np.random.seed(random_seed)
    cp.random.seed(random_seed)

    all_entities = set()
    for df in [train_triples, valid_triples, test_triples]:
        all_entities.update(df["head"].unique())
        all_entities.update(df["tail"].unique())

    print(f"Number of entities: {len(all_entities)}")

    missing = [e for e in all_entities if e not in node2emb]
    if missing:
        print(f"{len(missing)} entities missing in embeddings, filling with random vectors")
        emb_dim = next(iter(node2emb.values())).shape[0]
        for e in missing:
            node2emb[e] = (np.random.randn(emb_dim).astype(np.float32) * 0.1)

    entity_list = sorted(all_entities)
    entity_to_idx = {e: i for i, e in enumerate(entity_list)}

    emb_dim = len(next(iter(node2emb.values())))
    X_features = np.zeros((len(entity_list), emb_dim), dtype=np.float32)
    for e, idx in entity_to_idx.items():
        X_features[idx] = node2emb[e]

    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    X_features = cp.asarray(X_features, dtype=cp.float32)

    print(f"Feature matrix shape: {X_features.shape}")
    return entity_to_idx, X_features


# ============================================================
# Classifier data preparation
# ============================================================

def prepare_classifier_data(triples, entity_to_idx, X_features, relation_encoder):
    features = []
    labels = []

    for _, row in triples.iterrows():
        head_idx = entity_to_idx[row['head']]
        tail_idx = entity_to_idx[row['tail']]
        rel_label = relation_encoder.transform([row['relation']])[0]

        if isinstance(X_features, cp.ndarray):
            head_features = cp.asnumpy(X_features[head_idx])
            tail_features = cp.asnumpy(X_features[tail_idx])
        else:
            head_features = X_features[head_idx]
            tail_features = X_features[tail_idx]

        combined_features = np.concatenate([head_features, tail_features])
        features.append(combined_features)
        labels.append(rel_label)

    return np.array(features), np.array(labels)


def prepare_classifier_data_subtract(triples, entity_to_idx, X_features, relation_encoder):
    features = []
    labels = []

    for _, row in triples.iterrows():
        head_idx = entity_to_idx[row['head']]
        tail_idx = entity_to_idx[row['tail']]
        rel_label = relation_encoder.transform([row['relation']])[0]

        if isinstance(X_features, cp.ndarray):
            head_features = cp.asnumpy(X_features[head_idx])
            tail_features = cp.asnumpy(X_features[tail_idx])
        else:
            head_features = X_features[head_idx]
            tail_features = X_features[tail_idx]

        relation_features = head_features - tail_features
        features.append(relation_features)
        labels.append(rel_label)

    return np.array(features), np.array(labels)


# ============================================================
# Feature Translation
# ============================================================

def feature_translation(train_triples, valid_triples, test_triples,
                        entity_to_idx, X_features, relation_encoder):
    if isinstance(X_features, cp.ndarray):
        X_np = cp.asnumpy(X_features)
    else:
        X_np = X_features

    num_relations = len(relation_encoder.classes_)

    translation_vectors = np.zeros((num_relations, X_np.shape[1]))
    relation_counts = np.zeros(num_relations)

    for _, row in train_triples.iterrows():
        head_idx = entity_to_idx[row['head']]
        tail_idx = entity_to_idx[row['tail']]
        rel_idx = relation_encoder.transform([row['relation']])[0]

        translation_vectors[rel_idx] += (X_np[tail_idx] - X_np[head_idx])
        relation_counts[rel_idx] += 1

    for i in range(num_relations):
        if relation_counts[i] > 0:
            translation_vectors[i] /= relation_counts[i]

    def predict(head_idx, tail_idx):
        actual_translation = X_np[tail_idx] - X_np[head_idx]
        distances = []
        for trans_vec in translation_vectors:
            dist = np.linalg.norm(actual_translation - trans_vec)
            distances.append(dist)
        return np.argmin(distances)

    def predict_proba(head_indices, tail_indices):
        batch_size = len(head_indices)
        probabilities = np.zeros((batch_size, num_relations))

        for i, (head_idx, tail_idx) in enumerate(zip(head_indices, tail_indices)):
            actual_translation = X_np[tail_idx] - X_np[head_idx]
            distances = []
            for trans_vec in translation_vectors:
                dist = np.linalg.norm(actual_translation - trans_vec)
                distances.append(-dist)
            probabilities[i] = softmax(distances)

        return probabilities

    def evaluate(triples, name):
        preds = []
        truths = []

        for _, row in triples.iterrows():
            if row['head'] in entity_to_idx and row['tail'] in entity_to_idx:
                head_idx = entity_to_idx[row['head']]
                tail_idx = entity_to_idx[row['tail']]
                true_rel = relation_encoder.transform([row['relation']])[0]

                pred_rel = predict(head_idx, tail_idx)
                preds.append(pred_rel)
                truths.append(true_rel)

        acc = accuracy_score(truths, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        return acc

    print("Feature Translation results:")
    train_acc = evaluate(train_triples, "Train")
    valid_acc = evaluate(valid_triples, "Valid")
    test_acc = evaluate(test_triples, "Test")

    mrr = compute_translation_mrr(test_triples, entity_to_idx, X_features,
                                  relation_encoder, translation_vectors)
    hits_results = compute_translation_hits_at_k(test_triples, entity_to_idx, X_features,
                                                 relation_encoder, translation_vectors)

    print(f"Test MRR: {mrr:.4f}")
    for metric, value in hits_results.items():
        print(f"Test {metric}: {value:.4f}")

    return predict_proba, train_acc, valid_acc, test_acc, mrr, hits_results


def compute_translation_mrr(test_triples, entity_to_idx, X_features, relation_encoder, translation_vectors):
    if isinstance(X_features, cp.ndarray):
        X_np = cp.asnumpy(X_features)
    else:
        X_np = X_features

    reciprocal_ranks = []

    for _, row in test_triples.iterrows():
        if row['head'] in entity_to_idx and row['tail'] in entity_to_idx:
            head_idx = entity_to_idx[row['head']]
            tail_idx = entity_to_idx[row['tail']]
            true_rel = relation_encoder.transform([row['relation']])[0]

            actual_translation = X_np[tail_idx] - X_np[head_idx]

            distances = []
            for trans_vec in translation_vectors:
                dist = np.linalg.norm(actual_translation - trans_vec)
                distances.append(dist)

            scores = [-d for d in distances]
            sorted_indices = np.argsort(scores)[::-1]

            rank = np.where(sorted_indices == true_rel)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0


def compute_translation_hits_at_k(test_triples, entity_to_idx, X_features, relation_encoder,
                                  translation_vectors, k_values=[1, 3, 10]):
    if isinstance(X_features, cp.ndarray):
        X_np = cp.asnumpy(X_features)
    else:
        X_np = X_features

    hits = {k: 0 for k in k_values}
    total = 0

    for _, row in test_triples.iterrows():
        if row['head'] in entity_to_idx and row['tail'] in entity_to_idx:
            head_idx = entity_to_idx[row['head']]
            tail_idx = entity_to_idx[row['tail']]
            true_rel = relation_encoder.transform([row['relation']])[0]

            actual_translation = X_np[tail_idx] - X_np[head_idx]

            distances = []
            for trans_vec in translation_vectors:
                dist = np.linalg.norm(actual_translation - trans_vec)
                distances.append(dist)

            scores = [-d for d in distances]
            sorted_indices = np.argsort(scores)[::-1]

            for k in k_values:
                if true_rel in sorted_indices[:k]:
                    hits[k] += 1

            total += 1

    hits_at_k = {f"Hits@{k}": hits[k] / total if total > 0 else 0 for k in k_values}
    return hits_at_k


# ============================================================
# Classifier models
# ============================================================

def linear_feature_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test, X_features, num_relations, feature_method='concat', random_seed=28):
    lr = LogisticRegression(
        C=0.1,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        random_state=random_seed
    )
    lr.fit(X_train, y_train)

    def predict_proba(head_indices, tail_indices):
        features = []
        for head_idx, tail_idx in zip(head_indices, tail_indices):
            if isinstance(X_features, cp.ndarray):
                head_feat = cp.asnumpy(X_features[head_idx])
                tail_feat = cp.asnumpy(X_features[tail_idx])
            else:
                head_feat = X_features[head_idx]
                tail_feat = X_features[tail_idx]

            if feature_method == 'subtract':
                combined = head_feat - tail_feat
            else:
                combined = np.concatenate([head_feat, tail_feat])

            features.append(combined)
        raw_probas = lr.predict_proba(features)
        full = np.zeros((len(features), num_relations), dtype=np.float64)
        full[:, lr.classes_] = raw_probas
        return full

    train_pred = lr.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Logistic Regression ({feature_method})")
    print(f"Train Accuracy: {train_acc:.4f}")
    valid_pred = lr.predict(X_valid)
    valid_acc = accuracy_score(y_valid, valid_pred)
    print(f"Valid Accuracy: {valid_acc:.4f}")
    test_pred = lr.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    mrr, hits_results = compute_classifier_metrics(lr, X_test, y_test, num_relations)
    print(f"Test MRR: {mrr:.4f}")
    for metric, value in hits_results.items():
        print(f"Test {metric}: {value:.4f}")

    return predict_proba, train_acc, valid_acc, test_acc, mrr, hits_results


def random_forest_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test, X_features, num_relations, feature_method='concat', random_seed=28):
    rf = RandomForestClassifier(
        n_estimators=20,
        criterion='gini',
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=random_seed
    )
    rf.fit(X_train, y_train)
    print("Training complete")

    def predict_proba(head_indices, tail_indices):
        features = []
        for head_idx, tail_idx in zip(head_indices, tail_indices):
            if isinstance(X_features, cp.ndarray):
                head_feat = cp.asnumpy(X_features[head_idx])
                tail_feat = cp.asnumpy(X_features[tail_idx])
            else:
                head_feat = X_features[head_idx]
                tail_feat = X_features[tail_idx]

            if feature_method == 'subtract':
                combined = head_feat - tail_feat
            else:
                combined = np.concatenate([head_feat, tail_feat])

            features.append(combined)
        raw_probas = rf.predict_proba(features)
        full = np.zeros((len(features), num_relations), dtype=np.float64)
        full[:, rf.classes_] = raw_probas
        return full

    train_pred = rf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"Random Forest ({feature_method})")
    print(f"Train Accuracy: {train_acc:.4f}")
    valid_pred = rf.predict(X_valid)
    valid_acc = accuracy_score(y_valid, valid_pred)
    print(f"Valid Accuracy: {valid_acc:.4f}")
    test_pred = rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    mrr, hits_results = compute_classifier_metrics(rf, X_test, y_test, num_relations)
    print(f"Test MRR: {mrr:.4f}")
    for metric, value in hits_results.items():
        print(f"Test {metric}: {value:.4f}")

    return predict_proba, train_acc, valid_acc, test_acc, mrr, hits_results


def lgbm_classifier(X_train, y_train, X_valid, y_valid, X_test, y_test, X_features, num_relations, feature_method='concat', random_seed=28):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': num_relations,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'seed': random_seed,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'learning_rate': 0.1,
        'num_threads': min(os.cpu_count() or 1, 16)
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=300,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(20)]
    )
    print("Training complete")

    def predict_proba(head_indices, tail_indices):
        features = []
        for head_idx, tail_idx in zip(head_indices, tail_indices):
            if isinstance(X_features, cp.ndarray):
                head_feat = cp.asnumpy(X_features[head_idx])
                tail_feat = cp.asnumpy(X_features[tail_idx])
            else:
                head_feat = X_features[head_idx]
                tail_feat = X_features[tail_idx]

            if feature_method == 'subtract':
                combined = head_feat - tail_feat
            else:
                combined = np.concatenate([head_feat, tail_feat])
            features.append(combined)
        return model.predict(np.array(features))

    y_train_pred = np.argmax(model.predict(X_train), axis=1)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"LightGBM ({feature_method})")
    print(f"Train Accuracy: {train_acc:.4f}")
    y_valid_pred = np.argmax(model.predict(X_valid), axis=1)
    valid_acc = accuracy_score(y_valid, y_valid_pred)
    print(f"Valid Accuracy: {valid_acc:.4f}")
    y_test_pred = np.argmax(model.predict(X_test), axis=1)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_acc:.4f}")

    def compute_metrics(model, X, y_true):
        probas = model.predict(X)
        n_samples = len(y_true)
        reciprocal_ranks = []
        hits = {1: 0, 3: 0, 10: 0}
        for i, true_label in enumerate(y_true):
            scores = probas[i]
            sorted_indices = np.argsort(scores)[::-1]
            rank = np.where(sorted_indices == true_label)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)
            for k in [1, 3, 10]:
                if true_label in sorted_indices[:k]:
                    hits[k] += 1
        mrr = np.mean(reciprocal_ranks)
        hits_at_k = {f"Hits@{k}": hits[k] / n_samples for k in [1, 3, 10]}
        return mrr, hits_at_k

    mrr, hits_results = compute_metrics(model, X_test, y_test)
    print(f"Test MRR: {mrr:.4f}")
    for metric, value in hits_results.items():
        print(f"Test {metric}: {value:.4f}")

    return predict_proba, train_acc, valid_acc, test_acc, mrr, hits_results


# ============================================================
# Shared evaluation utilities
# ============================================================

def compute_classifier_metrics(model, X_test, y_test, num_relations=None, k_values=[1, 3, 10]):
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_test)
        model_classes = model.classes_
    else:
        try:
            probas = model.decision_function(X_test)
            model_classes = model.classes_
            if len(probas.shape) == 1:
                probas = np.column_stack([-probas, probas])
        except:
            preds = model.predict(X_test)
            model_classes = np.unique(y_test)
            num_classes = len(model_classes)
            probas = np.eye(num_classes)[preds]

    # If model was trained on a subset of relations, pad probas to full class space
    if num_relations is not None and probas.shape[1] < num_relations:
        full_probas = np.zeros((probas.shape[0], num_relations), dtype=probas.dtype)
        full_probas[:, model_classes] = probas
        probas = full_probas
        # Now sorted_indices correspond directly to global class indices
        use_global_labels = True
    else:
        use_global_labels = False

    reciprocal_ranks = []
    hits = {k: 0 for k in k_values}

    for i, true_label in enumerate(y_test):
        scores = probas[i]
        sorted_indices = np.argsort(scores)[::-1]
        if use_global_labels:
            sorted_labels = sorted_indices
        else:
            sorted_labels = model_classes[sorted_indices]
        rank = np.where(sorted_labels == true_label)[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)
        for k in k_values:
            if true_label in sorted_labels[:k]:
                hits[k] += 1

    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    hits_at_k = {f"Hits@{k}": hits[k] / len(y_test) for k in k_values}
    return mrr, hits_at_k


# ============================================================
# Conformal prediction
# ============================================================

def conformal_prediction(method_name, predict_proba_func,
                         calib_head_indices, calib_tail_indices, calib_true_labels,
                         test_head_indices, test_tail_indices, test_true_labels,
                         alpha_values, relation_encoder):
    print(f"\n{method_name.upper()} Conformal Prediction:")

    calib_probs = predict_proba_func(calib_head_indices, calib_tail_indices)
    test_probs = predict_proba_func(test_head_indices, test_tail_indices)
    results = {}
    for alpha in alpha_values:
        print(f"Alpha = {alpha}:")
        prediction_sets = generate_conformal_prediction_sets(
            calib_probs=calib_probs,
            test_probs=test_probs,
            calib_true_labels=calib_true_labels,
            alpha=alpha
        )
        eval_results = evaluate_conformal_prediction(prediction_sets, test_true_labels, alpha)
        results[alpha] = {
            'coverage': eval_results['marginal_coverage'],
            'avg_set_size': eval_results['average_set_size']
        }
        print(f"  Coverage: {eval_results['marginal_coverage']:.4f}")
        print(f"  Avg. Set Size: {eval_results['average_set_size']:.2f}")
    return results


def generate_conformal_prediction_sets(calib_probs=None, test_probs=None, calib_true_labels=None, alpha=0.1):
    calib_scores = 1 - calib_probs[np.arange(len(calib_true_labels)), calib_true_labels]
    n_calib = len(calib_scores)
    q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
    q_hat = np.quantile(calib_scores, q_level, method='higher')
    print(f"Calibration set size: {n_calib}, Quantile: {q_hat:.4f} (alpha={alpha})")

    prediction_sets = []
    for probs in test_probs:
        pred_set = set(np.where(probs >= 1 - q_hat)[0])
        prediction_sets.append(pred_set)
    return prediction_sets


def evaluate_conformal_prediction(prediction_sets, true_labels, alpha=0.1):
    coverage = np.mean([true_label in pred_set
                        for true_label, pred_set in zip(true_labels, prediction_sets)])
    avg_set_size = np.mean([len(pred_set) for pred_set in prediction_sets])
    return {'marginal_coverage': coverage, 'average_set_size': avg_set_size}


def prepare_indices_and_labels(triples_df, entity_to_idx, relation_encoder):
    heads = triples_df['head'].map(entity_to_idx)
    tails = triples_df['tail'].map(entity_to_idx)
    relations = triples_df['relation']

    valid_mask = heads.notna() & tails.notna()
    head_indices = heads[valid_mask].astype(int).values
    tail_indices = tails[valid_mask].astype(int).values
    true_labels = relation_encoder.transform(relations[valid_mask])

    return head_indices, tail_indices, true_labels


def setup_gpu_environment():
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    return pool


# ============================================================
# Main training entry point for a single version
# ============================================================

def run_baseline_version(version, model="roberta", aggregation="sum", random_seed=28):
    print("=" * 70)
    print(f"Baseline on {version} ({model}/{aggregation})")
    print("=" * 70)

    train_triples, valid_triples, test_triples, node2emb = load_fb237_data(
        version, model=model, aggregation=aggregation)

    print("Preparing feature matrix...")
    entity_to_idx, X_features = prepare_features_fb237(
        train_triples, valid_triples, test_triples, node2emb, random_seed
    )

    all_relations = sorted(set(train_triples['relation'].unique())
                          | set(valid_triples['relation'].unique())
                          | set(test_triples['relation'].unique()))
    R_train, relation_encoder, num_relations = create_sparse_relation_matrices(
        train_triples, entity_to_idx, all_relations=all_relations
    )
    print(f"Entities={X_features.shape[0]}, Feature dim={X_features.shape[1]}, Relations={num_relations}")

    # Prepare indices for conformal prediction
    calib_head_indices, calib_tail_indices, calib_true_labels = prepare_indices_and_labels(
        valid_triples, entity_to_idx, relation_encoder
    )
    test_head_indices, test_tail_indices, test_true_labels = prepare_indices_and_labels(
        test_triples, entity_to_idx, relation_encoder
    )

    alpha_values = [0.01, 0.05, 0.1]
    results = {}

    # ---- Feature Translation ----
    print("=" * 60)
    print("Feature Translation")
    t0 = time.time()
    ft_predict_proba, ft_train_acc, ft_valid_acc, ft_test_acc, ft_mrr, ft_hits = feature_translation(
        train_triples, valid_triples, test_triples,
        entity_to_idx, X_features, relation_encoder
    )
    ft_train_time = time.time() - t0
    ft_conformal = conformal_prediction(
        'Feature Translation', ft_predict_proba,
        calib_head_indices, calib_tail_indices, calib_true_labels,
        test_head_indices, test_tail_indices, test_true_labels,
        alpha_values, relation_encoder
    )
    results['feature_translation'] = {
        'train_acc': ft_train_acc, 'valid_acc': ft_valid_acc,
        'test_acc': ft_test_acc, 'mrr': ft_mrr, 'hits': ft_hits, 'conformal': ft_conformal,
        'train_time': ft_train_time
    }

    # ---- Classifiers with concat features ----
    for feature_method in ['concat', 'subtract']:
        if feature_method == 'concat':
            X_train, y_train = prepare_classifier_data(train_triples, entity_to_idx, X_features, relation_encoder)
            X_valid, y_valid = prepare_classifier_data(valid_triples, entity_to_idx, X_features, relation_encoder)
            X_test, y_test = prepare_classifier_data(test_triples, entity_to_idx, X_features, relation_encoder)
        else:
            X_train, y_train = prepare_classifier_data_subtract(train_triples, entity_to_idx, X_features, relation_encoder)
            X_valid, y_valid = prepare_classifier_data_subtract(valid_triples, entity_to_idx, X_features, relation_encoder)
            X_test, y_test = prepare_classifier_data_subtract(test_triples, entity_to_idx, X_features, relation_encoder)

        # Logistic Regression
        print("=" * 60)
        print(f"Logistic Regression ({feature_method})")
        t0 = time.time()
        lr_proba, lr_train_acc, lr_valid_acc, lr_test_acc, lr_mrr, lr_hits = linear_feature_classifier(
            X_train, y_train, X_valid, y_valid, X_test, y_test, X_features, num_relations, feature_method, random_seed
        )
        lr_train_time = time.time() - t0
        lr_conformal = conformal_prediction(
            f'LR_{feature_method}', lr_proba,
            calib_head_indices, calib_tail_indices, calib_true_labels,
            test_head_indices, test_tail_indices, test_true_labels,
            alpha_values, relation_encoder
        )
        results[f'lr_{feature_method}'] = {
            'train_acc': lr_train_acc, 'valid_acc': lr_valid_acc,
            'test_acc': lr_test_acc, 'mrr': lr_mrr, 'hits': lr_hits, 'conformal': lr_conformal,
            'train_time': lr_train_time
        }

        # Random Forest
        print("=" * 60)
        print(f"Random Forest ({feature_method})")
        t0 = time.time()
        rf_proba, rf_train_acc, rf_valid_acc, rf_test_acc, rf_mrr, rf_hits = random_forest_classifier(
            X_train, y_train, X_valid, y_valid, X_test, y_test, X_features, num_relations, feature_method, random_seed
        )
        rf_train_time = time.time() - t0
        rf_conformal = conformal_prediction(
            f'RF_{feature_method}', rf_proba,
            calib_head_indices, calib_tail_indices, calib_true_labels,
            test_head_indices, test_tail_indices, test_true_labels,
            alpha_values, relation_encoder
        )
        results[f'rf_{feature_method}'] = {
            'train_acc': rf_train_acc, 'valid_acc': rf_valid_acc,
            'test_acc': rf_test_acc, 'mrr': rf_mrr, 'hits': rf_hits, 'conformal': rf_conformal,
            'train_time': rf_train_time
        }

        # LightGBM
        print("=" * 60)
        print(f"LightGBM ({feature_method})")
        t0 = time.time()
        lgbm_proba, lgbm_train_acc, lgbm_valid_acc, lgbm_test_acc, lgbm_mrr, lgbm_hits = lgbm_classifier(
            X_train, y_train, X_valid, y_valid, X_test, y_test, X_features, num_relations, feature_method, random_seed
        )
        lgbm_train_time = time.time() - t0
        lgbm_conformal = conformal_prediction(
            f'LGBM_{feature_method}', lgbm_proba,
            calib_head_indices, calib_tail_indices, calib_true_labels,
            test_head_indices, test_tail_indices, test_true_labels,
            alpha_values, relation_encoder
        )
        results[f'lgbm_{feature_method}'] = {
            'train_acc': lgbm_train_acc, 'valid_acc': lgbm_valid_acc,
            'test_acc': lgbm_test_acc, 'mrr': lgbm_mrr, 'hits': lgbm_hits, 'conformal': lgbm_conformal,
            'train_time': lgbm_train_time
        }

    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None, help="Specific version (e.g. fb237_v1)")
    parser.add_argument("--all", action="store_true", help="Run on all 4 inductive versions")
    parser.add_argument("--model", type=str, default="roberta",
                        choices=["roberta", "llama3", "qwen"],
                        help="Which PLM embeddings to use")
    parser.add_argument("--aggregation", type=str, default="sum",
                        choices=["sum", "mean", "concat", "attn"],
                        help="Aggregation method used in generate_plm_embeddings.py")
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()

    if args.all:
        versions = VERSIONS
    elif args.version:
        versions = [args.version]
    else:
        print("Specify --version <name> or --all")
        return

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count())
    os.environ["OMP_NUM_THREADS"] = str(min(os.cpu_count() or 1, 16))

    pool = setup_gpu_environment()
    all_results = []

    try:
        for v in versions:
            result = run_baseline_version(v, model=args.model,
                                          aggregation=args.aggregation,
                                          random_seed=args.seed)
            all_results.append({'version': v, 'model': args.model,
                                'aggregation': args.aggregation, 'results': result})
            cp.get_default_memory_pool().free_all_blocks()
    finally:
        pool.free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()

    # Summary
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    for entry in all_results:
        v = entry['version']
        print(f"\n{v} ({entry['model']}/{entry['aggregation']}):")
        for method_name, r in entry['results'].items():
            print(f"  {method_name}: Acc={r['test_acc']:.4f}, MRR={r['mrr']:.4f}, "
                  f"Hits@1={r['hits']['Hits@1']:.4f}, "
                  f"Hits@3={r['hits']['Hits@3']:.4f}, "
                  f"Hits@10={r['hits']['Hits@10']:.4f}")

    # Save results to unified CSV (shared with IMC and kge-baseline)
    csv_path = os.path.join(os.path.dirname(__file__), "results", "fb237_unified_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    rows = []
    for entry in all_results:
        for method_name, r in entry['results'].items():
            row = {
                'method': method_name,
                'model': entry['model'],
                'aggregation': entry['aggregation'],
                'version': entry['version'],
                'train_time_s': round(r.get('train_time', 0), 1),
                'train_acc': round(r['train_acc'], 6),
                'valid_acc': round(r['valid_acc'], 6),
                'mrr': round(r['mrr'], 6),
                'hits_1': round(r['hits']['Hits@1'], 6),
                'hits_3': round(r['hits']['Hits@3'], 6),
                'hits_10': round(r['hits']['Hits@10'], 6),
            }
            for alpha in [0.01, 0.05, 0.1]:
                if alpha in r.get('conformal', {}):
                    cr = r['conformal'][alpha]
                    row[f'conformal_cov_a{alpha}'] = round(cr['coverage'], 6)
                    row[f'conformal_size_a{alpha}'] = round(cr['avg_set_size'], 4)
            rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        keys = df[['method', 'model', 'aggregation', 'version']].apply(tuple, axis=1).tolist()
        mask = existing[['method', 'model', 'aggregation', 'version']].apply(tuple, axis=1).isin(keys)
        existing = existing[~mask]
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    try:
        main()
        print("All baseline runs complete.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
