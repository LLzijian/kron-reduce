from __future__ import annotations


def assert_optimizer_lineage(train_ids, val_ids, test_ids) -> None:
    train, val, test = set(train_ids), set(val_ids), set(test_ids)
    if train & val or train & test or val & test:
        raise AssertionError("train/val/test lineage overlap")


def assert_fit_result_lineage(fit_result, allowed_train_ids, allowed_val_ids) -> None:
    if not set(fit_result.train_case_ids) <= set(allowed_train_ids):
        raise AssertionError("optimizer used an event outside the registered train role")
    if not set(fit_result.val_case_ids) <= set(allowed_val_ids):
        raise AssertionError("checkpoint selection used an event outside the registered val role")
