# Active working memory

# Working Memory System
class WorkingMemory:
    def __init__(self):
        """
        Initialize a working memory system.
        """
        self.memory = {}

    def store(self, key, value):
        """
        Store a value in memory.
        :param key: Identifier for the value.
        :param value: Value to store.
        """
        self.memory[key] = value

    def retrieve(self, key):
        """
        Retrieve a value from memory.
        :param key: Identifier for the value.
        :return: Retrieved value or None if key not found.
        """
        return self.memory.get(key)
