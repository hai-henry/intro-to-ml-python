import graphviz


def main():
    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph).render("output", format="pdf", cleanup=True)


if __name__ == "__main__":
    main()
