from coco_pipe.decoding._cache import make_feature_cache_key


def test_make_feature_cache_key_is_stable():
    train = ["s1", "s2"]
    test = ["s3"]
    prep = "p1"
    backbone = "b1"
    meta = {"task": "classify"}

    key1 = make_feature_cache_key(train, test, prep, backbone, meta)
    key2 = make_feature_cache_key(train, test, prep, backbone, meta)

    assert key1 == key2
    assert isinstance(key1, str)
    assert len(key1) == 64  # SHA256 length


def test_make_feature_cache_key_is_order_insensitive_by_default():
    train1 = ["s1", "s2"]
    train2 = ["s2", "s1"]
    test = ["s3"]
    prep = "p1"
    backbone = "b1"

    key1 = make_feature_cache_key(train1, test, prep, backbone)
    key2 = make_feature_cache_key(train2, test, prep, backbone)

    # Default is now order-insensitive
    assert key1 == key2


def test_make_feature_cache_key_can_be_order_sensitive():
    train1 = ["s1", "s2"]
    train2 = ["s2", "s1"]
    test = ["s3"]
    prep = "p1"
    backbone = "b1"

    key1 = make_feature_cache_key(train1, test, prep, backbone, sort_ids=False)
    key2 = make_feature_cache_key(train2, test, prep, backbone, sort_ids=False)

    # With sort_ids=False, order matters
    assert key1 != key2


def test_make_feature_cache_key_is_sensitive_to_fingerprints():
    train = ["s1"]
    test = ["s2"]

    key_base = make_feature_cache_key(train, test, "p1", "b1")
    key_diff_prep = make_feature_cache_key(train, test, "p2", "b1")
    key_diff_back = make_feature_cache_key(train, test, "p1", "b2")

    assert key_base != key_diff_prep
    assert key_base != key_diff_back


def test_make_feature_cache_key_is_sensitive_to_samples():
    prep = "p1"
    backbone = "b1"

    key1 = make_feature_cache_key(["s1"], ["s2"], prep, backbone)
    key2 = make_feature_cache_key(["s1", "s2"], ["s3"], prep, backbone)
    key3 = make_feature_cache_key(["s1"], ["s3"], prep, backbone)

    assert key1 != key2
    assert key1 != key3
    assert key2 != key3


def test_make_feature_cache_key_handles_extra_metadata():
    train = ["s1"]
    test = ["s2"]
    prep = "p1"
    backbone = "b1"

    key_no_meta = make_feature_cache_key(train, test, prep, backbone)
    key_meta1 = make_feature_cache_key(train, test, prep, backbone, {"time": 0})
    key_meta2 = make_feature_cache_key(train, test, prep, backbone, {"time": 1})

    assert key_no_meta != key_meta1
    assert key_meta1 != key_meta2


def test_make_feature_cache_key_converts_ids_to_strings():
    # Mixing types should work because they are converted to str
    train = [1, 2.0, "3"]
    test = [4]

    key1 = make_feature_cache_key(train, test, "p", "b")
    key2 = make_feature_cache_key(["1", "2.0", "3"], ["4"], "p", "b")

    assert key1 == key2


def test_make_feature_cache_key_extra_metadata_paths():
    """Verify both None and dict paths for coverage."""
    train, test = ["s1"], ["s2"]
    p, b = "p", "b"

    # Path 1: None (results in empty dict)
    key_none = make_feature_cache_key(train, test, p, b, extra_metadata=None)

    # Path 2: Explicit empty dict
    key_empty = make_feature_cache_key(train, test, p, b, extra_metadata={})

    # Path 3: Non-empty dict
    key_data = make_feature_cache_key(train, test, p, b, extra_metadata={"a": 1})

    assert key_none == key_empty
    assert key_none != key_data
