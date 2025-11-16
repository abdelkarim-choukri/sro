import traceback
try:
    import sro.claims.splitter as sp
    print('OK, splitter file=', sp.__file__)
except Exception as e:
    print('ERR', e)
    traceback.print_exc()
