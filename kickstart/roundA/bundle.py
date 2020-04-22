from collections import deque


class TrieNode:
    def __init__(self):
        self.passes = 0
        self.children = dict()


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.passes += 1


def helper(strings, noPick):
    trie = Trie()
    for string in strings:
        trie.insert(string)

    queue = deque()
    queue.append(trie.root)
    res = 0
    while queue:
        # res = queue[0].depth
        # groups = len(queue) // noPick
        node = queue.popleft()
        res+=node.passes//noPick
        for child in node.children.values():
            if child.passes >= noPick:
                queue.append(child)

    return res


if __name__ == '__main__':
    tests = int(input())
    for i in range(1, tests + 1):
        noString, noPick = [int(s) for s in input().split(" ")]
        strings = [input() for _ in range(noString)]
        print("Case #%s: %s" % (i, helper(strings, noPick)))
