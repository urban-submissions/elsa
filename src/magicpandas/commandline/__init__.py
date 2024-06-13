
"""
If it's an arg under magic.Commandline, it only shows when that commandline function is called.
If it's an arg found elsewhere, it shows whenever any commandline function is called.
If it's a commandline found anywhere, name discrepancies are resolved by adding the owner name.
Here, train and epoch both have arch as general args, but this is resolved by adding the owner name:
model.arch, epoch.arch
"""

# from magicpandas.commandline.commandline import CommandLine