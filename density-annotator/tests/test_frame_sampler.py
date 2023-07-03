import pytest
from annotator.exceptions import SamplingTypeError
from annotator.frame_sampler import get_frame_number_list, get_sampled_frame_number


def test_get_sampled_frame_number() -> None:
    total_frame_number = 100
    sample_rate = 10
    frame_number_list = get_sampled_frame_number(total_frame_number, sample_rate)
    assert len(frame_number_list) == 10
    for i, start in enumerate(range(1, total_frame_number, sample_rate)):
        assert start <= frame_number_list[i] < start + sample_rate


def test_get_frame_number_list() -> None:
    # random case
    total_frame_number = 100
    sample_rate = 10
    random_frame_number_list = get_frame_number_list(
        total_frame_number, "random", sample_rate
    )
    assert len(random_frame_number_list) == 10
    for i, start in enumerate(range(1, total_frame_number, sample_rate)):
        assert start <= random_frame_number_list[i] < start + sample_rate

    # fixed case
    fixed_frame_number_list = get_frame_number_list(
        total_frame_number, "fixed", sample_rate
    )
    expected = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert fixed_frame_number_list.sort() == expected.sort()

    # invalid sample type case
    with pytest.raises(SamplingTypeError):
        get_frame_number_list(total_frame_number, "invalid", sample_rate)
