def table_print(data, space, spacer=' | ', direction="<"):
    print(spacer.join(["{:{}{}}".format(d, direction, space) for d in data]))