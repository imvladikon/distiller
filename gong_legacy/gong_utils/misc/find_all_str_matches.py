import re
def find_all_str_matches(s, subs):
    """
    Finds "whole words only" matches of expressions in a given string
    :param s: The string in which to search for the given expressions in subs.
    :param subs: A list of expressions to search for in s.
    :return: All matches as spans (from, to) in s.
    """
    spans = []
    for p in subs:
        matches = re.finditer(p, s, re.IGNORECASE | re.MULTILINE)
        for m in matches:
            spans.append(m.span())
    return spans