# -*- coding: utf-8 -*-
from snipy.ilogging import getlogger

# function shortcuts
logg = getlogger('sflow')
info = logg.info
warn = logg.warn
debug = logg.debug
fatal = logg.critical
error = logg.error
exception = logg.exception

