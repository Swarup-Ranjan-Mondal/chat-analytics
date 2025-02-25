import networkx as nx
import matplotlib.pyplot as plt

def build_interaction_network(df):
    # Creates a directed network graph from chat interactions.
    G = nx.DiGraph()

    for i in range(len(df) - 1):
        sender = df.iloc[i]["user"]
        recipient = df.iloc[i + 1]["user"]

        # Avoid self-loops (same person sending consecutive messages)
        if sender != recipient:
            if G.has_edge(sender, recipient):
                G[sender][recipient]["weight"] += 1
            else:
                G.add_edge(sender, recipient, weight=1)
    return G

def visualize_network(G, st):
    # Generates and displays the network graph in Streamlit.
    fig, ax = plt.subplots(figsize=(10, 7))

    pos = nx.spring_layout(G, seed=42)  # Position nodes for visualization
    node_sizes = [G.degree(n) * 300 for n in G.nodes()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue",
            edge_color=edge_weights, edge_cmap=plt.cm.Blues, font_size=10, font_weight="bold", ax=ax)

    st.pyplot(fig)

def analyze_network(G, st):
    # Displays key network statistics in Streamlit.
    st.subheader("🔹 Network Analysis Summary")
    st.write(f"📌 **Total Users:** {G.number_of_nodes()}")
    st.write(f"📌 **Total Interactions:** {G.number_of_edges()}")
    
    degrees = dict(G.degree())
    most_active = max(degrees, key=degrees.get)
    st.write(f"👑 **Most Active User:** {most_active} ({degrees[most_active]} interactions)")

    centrality = nx.degree_centrality(G)
    top_central = max(centrality, key=centrality.get)
    st.write(f"📡 **Most Influential User:** {top_central} (Centrality Score: {centrality[top_central]:.2f})")
