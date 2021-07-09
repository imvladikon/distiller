def is_substring_in_group(c, group):
    for substring in group:
        if substring in c:
            return True
    return False


