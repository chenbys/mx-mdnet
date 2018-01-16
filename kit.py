from setting import config


def p(msg, level=0, prefix='@Chen ->', ):
    if level >= config.p_level:
        print prefix + str(msg)