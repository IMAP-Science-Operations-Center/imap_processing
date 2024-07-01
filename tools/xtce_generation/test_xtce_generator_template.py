"""Testing for xtce generator template."""

from unittest import mock


@mock.patch("tools.xtce_generation.")
def test_main():
    """Testing base main function."""
    test_args = [
        "test",
        "sWe",
        "test_file",
        "ana",
        "1",
        "gav",
        "2",
        "meg",
        "3",
        "Q",
        "N",
    ]
    # expected_result = {"ana": 1, "gav": 2, "meg": 3}
    input_values = iter(test_args)
    with mock.patch.object("builtins.input", lambda _: next(input_values)):
        pass


# @pytest.mark.parametrize(
#     "arguments,final",
#     [(
#         # Command line arguments
#         ["test", "sWe", "test_file", "ana", "1", "gav", "2", "meg", "3", "Q", "N"],
#         # Resulting dictionary
#         {"ana": 1, "gav": 2, "meg": 3}
#     )]
#     )
# def test_editing_packets_dictionary(mocked_input):
#     result = main(arguments)
#     assert result == final
