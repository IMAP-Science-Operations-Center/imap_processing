"""Testing for xtce generator template."""


# @mock.patch("tools.xtce_generation.xtce_generator_template.main")
# @mock.patch("builtins.input", side_effect=["mocked_instrument_name"])
# def test_main(mock_main, mock_input):
#     """Testing base main function."""
#     # Testing lowercase instrument_name, quit, and "no" for editing
#     test_args = [
#         "test",
#         "sWe",
#         "test_file",
#         "ana",
#         "1",
#         "gav",
#         "2",
#         "meg",
#         "3",
#         "Q",
#         "N",
#     ]
#     expected_result = {"ana": 1, "gav": 2, "meg": 3}
#     with mock.patch.object(sys, "argv", test_args):
#         assert main() == expected_result
#
#     # INPUTS TO TEST AGAINST!
#     # Testing editing option 1
#     test_args_2 = [
#         "Not",
#         "Not again",
#         "codice",
#         "file",
#         "Daralynn",
#         "1",
#         "Isaiah",
#         "2",
#         "Q",
#         "Y",
#         "1",
#         "Ana",
#         "Isaiah",
#         "Rashad",
#         "q",
#     ]
#     expected_result_2 = {"Daralynn": 1, "Rashad": 2}
#
#     # Testing editing option 2
#     test_args_3 = [
#         "glows",
#         "Fizzile",
#         "JID",
#         "1",
#         "Kendrick",
#         "2",
#         "J. Cole" "3" "Q",
#         "Y",
#         "2",
#         "Gav",
#         "Kendrick",
#         "1",
#     ]
#     expected_result_3 = {"JID": 1, "Kendrick": 1, "J. Cole": 3}
#
#     # Testing editing option 3
#     test_args_4 = [
#         "hi",
#         "fancy_file",
#         "spider",
#         "1",
#         "mantis",
#         "2",
#         "cricket",
#         "3",
#         "Q",
#         "Y" "3",
#         "not name" "cricket",
#     ]
#     expected_result_4 = {"spider": 1, "mantis": 2}
#
#     # Testing editing option 4
#     test_args_5 = [
#         "hit",
#         "not_fancy_file",
#         "Ash",
#         "1",
#         "Willow",
#         "2",
#         "Bradford Pear",
#         "3",
#         "Q",
#         "Y",
#         "4",
#         "Redwood",
#         "4",
#         "",
#     ]
#     expected_result_5 = {"Ash": 1, "Willow": 2, "Bradford Pear": 3, "Redwood": 4}
#
#     # Testing editing option Q
#     test_args_6 = [
#         "lo",
#         "Apple",
#         "1",
#         "Microsoft",
#         "2",
#         "Linux",
#         "3",
#         "Q",
#         "Y",
#         "Q",
#     ]
#     expected_result_6 = {"Apple": 1, "Microsoft": 2, "Linux": 3}
