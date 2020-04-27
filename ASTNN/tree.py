class ASTNode(object):
    def __init__(self, node):
        self.node = node
        # self.vocab = word_map
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        # self.index = self.token_to_index(self.token)
        self.children = self.add_children()

    def is_leaf(self):
        if self.is_str:
            return True
        return len(list(self.node.get_children())) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = str(self.node.kind)
        token = name
        is_name = False
        if self.is_leaf():
            token = name
        else:
            if name == 'CursorKind.TYPEDEF_DECL':
                token = str(self.node.spelling)
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token

    # def token_to_index(self, token):
    #     self.index = self.vocab[token].index if token in self.vocab else MAX_TOKENS
    #     return self.index

    # def get_index(self):
    #     return self.index

    def add_children(self):
        if self.is_str:
            return []
        children = self.node.get_children()
        #if self.token in ['CursorKind.FUNCTION_DECL', 'CursorKind.IF_STMT', 'CursorKind.WHILE_STMT', 'CursorKind.DO_STMT']:
            #return [ASTNode(list(self.node.get_children()))]
        #elif self.token == 'CursorKind.FOR_STMT':
            #return [ASTNode(children[c][1]) for c in range(0, len(children)-1)]
        #else:
        return [ASTNode(child) for child in list(children)]

    def children(self):
        return self.children
    #     if self.is_str:
    #         return []
    #     return [ASTNode(child) for _, child in self.node.children() if child.]


class SingleNode(ASTNode):
    def __init__(self, node):
        self.node = node
        self.is_str = isinstance(self.node, str)
        self.token = self.get_token()
        self.children = []

    def is_leaf(self):
        if self.is_str:
            return True
        return len(list(self.node.get_children())) == 0

    def get_token(self, lower=True):
        if self.is_str:
            return self.node
        name = str(self.node.kind)
        token = name
        is_name = False
        if self.is_leaf():
            token = name
        else:
            if name == 'CursorKind.TYPEDEF_DECL':
                token = str(self.node.spelling)
        if token is None:
            token = name
        if lower and is_name:
            token = token.lower()
        return token