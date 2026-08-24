class Twofish:
    """Stub replacement — satisfies pyjks' dependency without a C compiler.
    Only breaks if something reads a legacy 'UBER' Java Keystore file,
    which Hopsworks' Feature Store client never does."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("twofish stub: unused for this project")
    def encrypt(self, *args, **kwargs):
        raise NotImplementedError("twofish stub: unused")
    def decrypt(self, *args, **kwargs):
        raise NotImplementedError("twofish stub: unused")