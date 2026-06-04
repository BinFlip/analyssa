//! Cycle detection algorithms for directed graphs.
//!
//! This module provides algorithms to detect and find cycles in directed graphs.
//! Cycle detection is essential for:
//!
//! - Validating that dependency graphs are acyclic (DAGs)
//! - Detecting recursive call patterns in call graphs
//! - Identifying loops in control flow graphs
//!
//! # Algorithm
//!
//! Both `has_cycle` and `find_cycle` use depth-first search with a two-vector
//! tracking scheme:
//!
//! - **visited**: Nodes that have been encountered during DFS are marked visited
//!   when first entered, preventing redundant exploration.
//! - **in_stack (recursion stack)**: Nodes currently on the DFS recursion path.
//!   A node is added to the stack when entered and removed when its entire
//!   subtree has been processed.
//!
//! A back edge to a node still on the recursion stack indicates a cycle:
//!
//! - `has_cycle` returns `true` immediately upon detecting a back edge, without
//!   reconstructing the cycle path.
//! - `find_cycle` extends the detection to reconstruct the cycle by capturing
//!   the portion of the recursion stack from the target node to the current node,
//!   returning it as a closed path (first node == last node).

use crate::graph::{NodeId, Successors};

/// Checks if a directed graph contains any cycles reachable from the start node.
///
/// This function uses depth-first search with a recursion stack to detect
/// back edges, which indicate cycles. It only considers nodes reachable
/// from the start node.
///
/// # Arguments
///
/// * `graph` - The graph to check for cycles
/// * `start` - The starting node for the search
///
/// # Returns
///
/// `true` if a cycle is found, `false` otherwise.
///
/// # Complexity
///
/// - Time: O(V + E) where V is the number of vertices and E is the number of edges
/// - Space: O(V) for the visited and recursion stack sets
///
/// # Examples
///
/// ```rust,ignore
/// use analyssa::graph::{DirectedGraph, NodeId, algorithms::has_cycle};
///
/// // Acyclic graph: A -> B -> C
/// let mut dag: DirectedGraph<(), ()> = DirectedGraph::new();
/// let a = dag.add_node(());
/// let b = dag.add_node(());
/// let c = dag.add_node(());
/// dag.add_edge(a, b, ());
/// dag.add_edge(b, c, ());
///
/// assert!(!has_cycle(&dag, a));
///
/// // Cyclic graph: A -> B -> C -> A
/// let mut cyclic: DirectedGraph<(), ()> = DirectedGraph::new();
/// let x = cyclic.add_node(());
/// let y = cyclic.add_node(());
/// let z = cyclic.add_node(());
/// cyclic.add_edge(x, y, ());
/// cyclic.add_edge(y, z, ());
/// cyclic.add_edge(z, x, ());
///
/// assert!(has_cycle(&cyclic, x));
/// ```
pub fn has_cycle<G: Successors>(graph: &G, start: NodeId) -> bool {
    let node_count = graph.node_count();
    if start.index() >= node_count {
        return false;
    }

    let mut visited = vec![false; node_count];
    let mut in_stack = vec![false; node_count];

    has_cycle_dfs(graph, start, &mut visited, &mut in_stack)
}

/// Iterative helper for cycle detection.
///
/// Uses an explicit work stack (one frame per gray node, carrying that node's
/// remaining successors) rather than call-stack recursion, so it cannot
/// overflow the stack on deep or long-chain graphs.
fn has_cycle_dfs<G: Successors>(
    graph: &G,
    start: NodeId,
    visited: &mut [bool],
    in_stack: &mut [bool],
) -> bool {
    let start_idx = start.index();
    if visited.get(start_idx).copied().unwrap_or(false) {
        return false;
    }
    if let Some(slot) = visited.get_mut(start_idx) {
        *slot = true;
    }
    if let Some(slot) = in_stack.get_mut(start_idx) {
        *slot = true;
    }

    // Each frame is a gray node and an iterator over its not-yet-visited
    // successors. The successor type is opaque, so it is materialized into a
    // `Vec` per frame.
    let mut stack: Vec<(NodeId, std::vec::IntoIter<NodeId>)> = vec![(
        start,
        graph.successors(start).collect::<Vec<_>>().into_iter(),
    )];

    while !stack.is_empty() {
        let next = match stack.last_mut() {
            Some(frame) => frame.1.next(),
            None => break,
        };
        match next {
            Some(successor) => {
                let s_idx = successor.index();
                if in_stack.get(s_idx).copied().unwrap_or(false) {
                    // Back edge to a node on the current path - cycle detected.
                    return true;
                }
                if visited.get(s_idx).copied().unwrap_or(false) {
                    continue;
                }
                if let Some(slot) = visited.get_mut(s_idx) {
                    *slot = true;
                }
                if let Some(slot) = in_stack.get_mut(s_idx) {
                    *slot = true;
                }
                stack.push((
                    successor,
                    graph.successors(successor).collect::<Vec<_>>().into_iter(),
                ));
            }
            None => {
                // Successors exhausted: leave the node (back to black).
                if let Some((node, _)) = stack.pop() {
                    if let Some(slot) = in_stack.get_mut(node.index()) {
                        *slot = false;
                    }
                }
            }
        }
    }
    false
}

/// Finds a cycle in a directed graph if one exists, starting from the given node.
///
/// If a cycle is found, returns a vector of nodes forming the cycle (starting
/// and ending with the same node). If no cycle is found, returns `None`.
///
/// # Arguments
///
/// * `graph` - The graph to search for cycles
/// * `start` - The starting node for the search
///
/// # Returns
///
/// `Some(Vec<NodeId>)` containing the cycle path if found, `None` otherwise.
/// The cycle path starts and ends with the same node.
///
/// # Complexity
///
/// - Time: O(V + E)
/// - Space: O(V)
///
/// # Examples
///
/// ```rust,ignore
/// use analyssa::graph::{DirectedGraph, NodeId, algorithms::find_cycle};
///
/// // Cyclic graph: A -> B -> C -> A
/// let mut graph: DirectedGraph<char, ()> = DirectedGraph::new();
/// let a = graph.add_node('A');
/// let b = graph.add_node('B');
/// let c = graph.add_node('C');
/// graph.add_edge(a, b, ());
/// graph.add_edge(b, c, ());
/// graph.add_edge(c, a, ());
///
/// let cycle = find_cycle(&graph, a);
/// assert!(cycle.is_some());
///
/// let cycle_nodes = cycle.unwrap();
/// assert!(cycle_nodes.len() >= 3); // At least 3 nodes in the cycle
/// assert_eq!(cycle_nodes.first(), cycle_nodes.last()); // Forms a cycle
/// ```
pub fn find_cycle<G: Successors>(graph: &G, start: NodeId) -> Option<Vec<NodeId>> {
    let node_count = graph.node_count();
    if start.index() >= node_count {
        return None;
    }

    let mut visited = vec![false; node_count];
    let mut in_stack = vec![false; node_count];
    let mut path = Vec::new();

    find_cycle_dfs(graph, start, &mut visited, &mut in_stack, &mut path)
}

/// Iterative helper for finding a cycle.
///
/// Like [`has_cycle_dfs`], this uses an explicit work stack instead of
/// recursion. `path` mirrors the gray nodes on the current DFS path, so a back
/// edge can be reconstructed into a closed cycle.
fn find_cycle_dfs<G: Successors>(
    graph: &G,
    start: NodeId,
    visited: &mut [bool],
    in_stack: &mut [bool],
    path: &mut Vec<NodeId>,
) -> Option<Vec<NodeId>> {
    let start_idx = start.index();
    if visited.get(start_idx).copied().unwrap_or(false) {
        return None;
    }
    if let Some(slot) = visited.get_mut(start_idx) {
        *slot = true;
    }
    if let Some(slot) = in_stack.get_mut(start_idx) {
        *slot = true;
    }
    path.push(start);

    let mut stack: Vec<(NodeId, std::vec::IntoIter<NodeId>)> = vec![(
        start,
        graph.successors(start).collect::<Vec<_>>().into_iter(),
    )];

    while !stack.is_empty() {
        let next = match stack.last_mut() {
            Some(frame) => frame.1.next(),
            None => break,
        };
        match next {
            Some(successor) => {
                let s_idx = successor.index();
                if in_stack.get(s_idx).copied().unwrap_or(false) {
                    // Back edge - extract the cycle from `path`.
                    let cycle_start_pos = path.iter().position(|&n| n == successor)?;
                    let mut cycle: Vec<NodeId> = path.get(cycle_start_pos..)?.to_vec();
                    cycle.push(successor); // Close the cycle.
                    return Some(cycle);
                }
                if visited.get(s_idx).copied().unwrap_or(false) {
                    continue;
                }
                if let Some(slot) = visited.get_mut(s_idx) {
                    *slot = true;
                }
                if let Some(slot) = in_stack.get_mut(s_idx) {
                    *slot = true;
                }
                path.push(successor);
                stack.push((
                    successor,
                    graph.successors(successor).collect::<Vec<_>>().into_iter(),
                ));
            }
            None => {
                if let Some((node, _)) = stack.pop() {
                    if let Some(slot) = in_stack.get_mut(node.index()) {
                        *slot = false;
                    }
                    path.pop();
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::graph::{
        algorithms::cycles::{find_cycle, has_cycle},
        DirectedGraph, NodeId,
    };

    fn create_linear_graph() -> DirectedGraph<'static, &'static str, ()> {
        let mut graph = DirectedGraph::new();
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph
    }

    fn create_diamond_graph() -> DirectedGraph<'static, &'static str, ()> {
        let mut graph = DirectedGraph::new();
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(a, c, ()).unwrap();
        graph.add_edge(b, d, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph
    }

    fn create_simple_cycle() -> DirectedGraph<'static, &'static str, ()> {
        let mut graph = DirectedGraph::new();
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, a, ()).unwrap();
        graph
    }

    fn create_self_loop() -> DirectedGraph<'static, &'static str, ()> {
        let mut graph = DirectedGraph::new();
        let a = graph.add_node("A");
        graph.add_edge(a, a, ()).unwrap();
        graph
    }

    fn create_complex_with_cycle() -> DirectedGraph<'static, &'static str, ()> {
        // A -> B -> C -> D
        //      ^       |
        //      +-------+
        let mut graph = DirectedGraph::new();
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, b, ()).unwrap();
        graph
    }

    #[test]
    fn test_has_cycle_linear() {
        let graph = create_linear_graph();
        assert!(!has_cycle(&graph, NodeId::new(0)));
    }

    #[test]
    fn test_has_cycle_diamond() {
        let graph = create_diamond_graph();
        assert!(!has_cycle(&graph, NodeId::new(0)));
    }

    #[test]
    fn test_has_cycle_simple_cycle() {
        let graph = create_simple_cycle();
        assert!(has_cycle(&graph, NodeId::new(0)));
    }

    #[test]
    fn test_has_cycle_self_loop() {
        let graph = create_self_loop();
        assert!(has_cycle(&graph, NodeId::new(0)));
    }

    #[test]
    fn test_has_cycle_complex() {
        let graph = create_complex_with_cycle();
        assert!(has_cycle(&graph, NodeId::new(0)));
    }

    #[test]
    fn test_has_cycle_single_node_no_loop() {
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        let a = graph.add_node(());
        assert!(!has_cycle(&graph, a));
    }

    #[test]
    fn test_has_cycle_two_separate_cycles() {
        // Two separate cycles not connected
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");
        let d = graph.add_node("D");

        // Cycle 1: A <-> B
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, a, ()).unwrap();

        // Cycle 2: C <-> D (disconnected from A, B)
        graph.add_edge(c, d, ()).unwrap();
        graph.add_edge(d, c, ()).unwrap();

        // Starting from A should find cycle in first component
        assert!(has_cycle(&graph, a));

        // Starting from C should find cycle in second component
        assert!(has_cycle(&graph, c));
    }

    #[test]
    fn test_find_cycle_linear() {
        let graph = create_linear_graph();
        assert!(find_cycle(&graph, NodeId::new(0)).is_none());
    }

    #[test]
    fn test_find_cycle_diamond() {
        let graph = create_diamond_graph();
        assert!(find_cycle(&graph, NodeId::new(0)).is_none());
    }

    #[test]
    fn test_find_cycle_simple_cycle() {
        let graph = create_simple_cycle();
        let cycle = find_cycle(&graph, NodeId::new(0));

        assert!(cycle.is_some());
        let cycle = cycle.unwrap();

        // Cycle should form a loop (first == last)
        assert_eq!(cycle.first(), cycle.last());

        // Should have at least 3 nodes in a triangle cycle plus the closing node
        assert!(cycle.len() >= 3);
    }

    #[test]
    fn test_find_cycle_self_loop() {
        let graph = create_self_loop();
        let cycle = find_cycle(&graph, NodeId::new(0));

        assert!(cycle.is_some());
        let cycle = cycle.unwrap();

        // Self loop: [A, A]
        assert_eq!(cycle.len(), 2);
        assert_eq!(cycle[0], cycle[1]);
    }

    #[test]
    fn test_find_cycle_complex() {
        let graph = create_complex_with_cycle();
        let cycle = find_cycle(&graph, NodeId::new(0));

        assert!(cycle.is_some());
        let cycle = cycle.unwrap();

        // Cycle B -> C -> D -> B
        assert_eq!(cycle.first(), cycle.last());
    }

    #[test]
    fn test_find_cycle_returns_valid_path() {
        let graph = create_simple_cycle();
        let cycle = find_cycle(&graph, NodeId::new(0)).unwrap();

        // Verify the path is valid: each node connects to the next
        for i in 0..cycle.len() - 1 {
            let current = cycle[i];
            let next = cycle[i + 1];
            let successors: Vec<NodeId> = graph.successors(current).collect();
            assert!(
                successors.contains(&next),
                "Invalid cycle path: no edge from {:?} to {:?}",
                current,
                next
            );
        }
    }

    #[test]
    fn test_has_cycle_deep_linear_chain_is_iterative() {
        // A long acyclic chain would overflow a recursive DFS; the iterative
        // implementation must handle it without blowing the stack.
        let mut graph: DirectedGraph<(), ()> = DirectedGraph::new();
        let mut nodes = Vec::new();
        for _ in 0..10_000 {
            nodes.push(graph.add_node(()));
        }
        for window in nodes.windows(2) {
            graph.add_edge(window[0], window[1], ()).unwrap();
        }
        assert!(!has_cycle(&graph, nodes[0]));
        assert!(find_cycle(&graph, nodes[0]).is_none());

        // Closing the chain into a giant cycle must still be detected.
        graph
            .add_edge(nodes[nodes.len() - 1], nodes[0], ())
            .unwrap();
        assert!(has_cycle(&graph, nodes[0]));
        let cycle = find_cycle(&graph, nodes[0]).unwrap();
        assert_eq!(cycle.first(), cycle.last());
    }

    #[test]
    fn test_find_cycle_disconnected_cycle() {
        // Entry point not in the cycle
        let mut graph: DirectedGraph<&str, ()> = DirectedGraph::new();
        let entry = graph.add_node("Entry");
        let a = graph.add_node("A");
        let b = graph.add_node("B");
        let c = graph.add_node("C");

        graph.add_edge(entry, a, ()).unwrap();
        graph.add_edge(a, b, ()).unwrap();
        graph.add_edge(b, c, ()).unwrap();
        graph.add_edge(c, a, ()).unwrap(); // Cycle: A -> B -> C -> A

        let cycle = find_cycle(&graph, entry);
        assert!(cycle.is_some());
    }
}
