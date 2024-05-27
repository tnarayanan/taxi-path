# taxi-path

## IP problem formulation

Let there be $n$ nodes $x_1, \dots, x_n$. Each node has an associated decision variable $y_i$ that corresponds to whether it should be in the path or not. The path starts at node $s$ and ends at node $t$. Let $c_i$ represent whether or not a node is claimed.

**Objective:** $\min {???}$

**Constraints:**

* $y_s \le 1$
  
  $y_s \ge 1$
  
  * The start node must be in the path
  
* $y_t \le 1$
  
  $y_t \ge 1$
  
  * The end node must be in the path
  
* $y_i \le 0 \hspace{1cm} i \in [n] \setminus \{s, t\}$
  
  $y_i \ge 1 \hspace{1cm} i \in [n] \setminus \{s, t\}$
  
  * All nodes must either be in or not in the path
  
* $\sum_{i \in [n]} c_i y_i \le 0$
  
  $\sum_{i \in [n]} c_i y_i \ge 0$
  
  * No claimed node can be on the path
  
* $\sum_{(s,j) \in E} y_j \le 1$
  
  $\sum_{(s,j) \in E} y_j \ge 1$
  
  * The number of nodes connected to the start that are in the path must be 1 (meaning the start is at an endpoint of the path)
  
* $\sum_{(t,j) \in E} y_j \le 1$
  
  $\sum_{(t,j) \in E} y_j \ge 1$
  
  * The number of nodes connected to the end that are in the path must be 1 (meaning the start is at an endpoint of the path)
  
* $\sum_{(i,j) \in E} y_j \le 2y_i + M(1-y_i) \hspace{1cm} i \in [n] \setminus \{s, t\}$

  $\sum_{(i,j) \in E} y_j \ge 2y_i - M(1-y_i) \hspace{1cm} i \in [n] \setminus \{s, t\}$

  * Any node in the path that is not the start or end node must have exactly 2 neighbors

