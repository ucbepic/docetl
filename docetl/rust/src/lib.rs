use pyo3::prelude::*;
use ndarray::{Array2, Array1, Axis};
use std::collections::{HashSet};
use pyo3::types::{PyDict, PyList};
use pyo3::Python;
use pyo3::types::PyModule;

#[derive(Debug, Clone)]
struct ComparisonPair {
    i: usize,
    j: usize,
    similarity: f64,
}

#[derive(Debug, Clone)]
struct BlockingRule {
    rule_type: String,
    key1: String,
    key2: String,
}

#[pyclass]
pub struct FastResolver {
    #[pyo3(get, set)]
    pub blocking_threshold: f64,
    parent: Vec<usize>,
    size: Vec<usize>,
    clusters: Vec<HashSet<usize>>,
    processed_pairs: HashSet<(usize, usize)>,
    blocking_rules: Vec<BlockingRule>,
}

#[pymethods]
impl FastResolver {
    #[new]
    fn new(blocking_threshold: f64) -> Self {
        FastResolver {
            blocking_threshold,
            parent: Vec::new(),
            size: Vec::new(),
            clusters: Vec::new(),
            processed_pairs: HashSet::new(),
            blocking_rules: Vec::new(),
        }
    }

    #[staticmethod]
    fn compute_similarity_matrix(embeddings: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let n = embeddings.len();
        let n_features = embeddings[0].len();
        
        // Convert to ndarray more efficiently using one allocation
        let embedding_data: Vec<f64> = embeddings.into_iter().flatten().collect();
        let embedding_matrix = Array2::from_shape_vec((n, n_features), embedding_data)
            .expect("Shape mismatch in embedding conversion");
        
        // Compute norms using axis operation
        let norms: Array1<f64> = embedding_matrix.map_axis(Axis(1), |row| {
            (row.dot(&row)).sqrt()
        });
        
        // Compute similarity matrix directly
        let dot_products = embedding_matrix.dot(&embedding_matrix.t());
        let norms_matrix = &norms.view().into_shape((n, 1)).unwrap() 
            * &norms.view().into_shape((1, n)).unwrap();
        
        // Divide element-wise and convert to Vec<Vec>
        let similarity = &dot_products / &norms_matrix;
        similarity.outer_iter()
            .map(|row| row.to_vec())
            .collect()
    }

    fn add_contains_rule(&mut self, key1: String, key2: String) -> PyResult<()> {
        self.blocking_rules.push(BlockingRule {
            rule_type: "contains".to_string(),
            key1,
            key2,
        });
        Ok(())
    }

    fn add_contained_in_rule(&mut self, key1: String, key2: String) -> PyResult<()> {
        self.blocking_rules.push(BlockingRule {
            rule_type: "contained_in".to_string(),
            key1,
            key2,
        });
        Ok(())
    }

    fn add_equals_rule(&mut self, key1: String, key2: String) -> PyResult<()> {
        self.blocking_rules.push(BlockingRule {
            rule_type: "equals".to_string(),
            key1,
            key2,
        });
        Ok(())
    }

    fn check_blocking_rules(&self, item1: &PyDict, item2: &PyDict) -> PyResult<bool> {
        for rule in &self.blocking_rules {
            let val1 = match item1.get_item(&rule.key1) {
                Some(v) => v.to_string().to_lowercase(),
                None => continue,
            };
            let val2 = match item2.get_item(&rule.key2) {
                Some(v) => v.to_string().to_lowercase(),
                None => continue,
            };

            match rule.rule_type.as_str() {
                "contains" => {
                    if val1.contains(&val2) {
                        return Ok(true);
                    }
                }
                "contained_in" => {
                    if val2.contains(&val1) {
                        return Ok(true);
                    }
                }
                "equals" => {
                    if val1 == val2 {
                        return Ok(true);
                    }
                }
                _ => continue,
            }
        }
        Ok(false)
    }

    fn process_items_with_rules<'py>(
        &mut self,
        _py: Python<'py>,
        items: &'py PyList,
    ) -> PyResult<Vec<(usize, usize)>> {
        let n_samples = items.len();
        let mut blocking_pairs = Vec::new();

        // Skip if no blocking rules
        if self.blocking_rules.is_empty() {
            return Ok(blocking_pairs);
        }

        // Check each pair against blocking rules
        for i in 0..n_samples {
            for j in (i+1)..n_samples {
                let item1 = items.get_item(i)?.downcast::<PyDict>()?;
                let item2 = items.get_item(j)?.downcast::<PyDict>()?;

                if self.check_blocking_rules(item1, item2)? {
                    // Only add if not already in same cluster and not processed
                    let root1 = self.find_cluster(i);
                    let root2 = self.find_cluster(j);
                    if root1 != root2 && !self.is_processed(i, j) {
                        blocking_pairs.push((i, j));
                    }
                }
            }
        }

        Ok(blocking_pairs)
    }

    fn process_embeddings(
        &mut self,
        embeddings: Vec<Vec<f64>>,
        items: Option<&PyList>,
    ) -> PyResult<Vec<(usize, usize)>> {
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }
        if !embeddings.iter().all(|v| v.len() == embeddings[0].len()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "All embeddings must have the same dimension"
            ));
        }
        Python::with_gil(|py| {
            let sys = PyModule::import(py, "sys")?;
            let stdout = sys.getattr("stdout")?;
            
            let n_samples = embeddings.len();
            stdout.call_method1("write", (format!("Processing embeddings for {} samples...\n", n_samples),))?;
            
            // Initialize union-find data structures
            self.parent = (0..n_samples).collect();
            self.size = vec![1; n_samples];
            self.clusters = vec![HashSet::new(); n_samples];
            for i in 0..n_samples {
                self.clusters[i].insert(i);
            }
            self.processed_pairs.clear();

            // Get pairs from embeddings
            stdout.call_method1("write", ("Computing similarity matrix...\n".to_string(),))?;
            let mut all_pairs = Vec::new();
            
            // Add embedding-based pairs
            let mut pairs = Vec::new();
            let similarity_matrix = Self::compute_similarity_matrix(embeddings);
            
            stdout.call_method1("write", ("Finding pairs above threshold...\n".to_string(),))?;
            for i in 0..n_samples {
                for j in (i+1)..n_samples {
                    let similarity = similarity_matrix[i][j];
                    if similarity >= self.blocking_threshold {
                        pairs.push(ComparisonPair { i, j, similarity });
                    }
                }
            }
            
            stdout.call_method1("write", 
                (format!("Found {} pairs above threshold {}\n", pairs.len(), self.blocking_threshold),))?;
            
            // Sort by similarity descending
            pairs.sort_unstable_by(|a, b| {
                b.similarity.partial_cmp(&a.similarity).unwrap()
            });
            
            // Convert to (i,j) pairs and add to all_pairs
            all_pairs.extend(pairs.into_iter().map(|pair| (pair.i, pair.j)));

            // Add blocking rule pairs if items were provided
            if let Some(items_list) = items {
                stdout.call_method1("write", ("Applying blocking rules...\n".to_string(),))?;
                let blocking_pairs = self.process_items_with_rules(py, items_list)?;
                stdout.call_method1("write", 
                    (format!("Found {} additional pairs from blocking rules\n", blocking_pairs.len()),))?;
                all_pairs.extend(blocking_pairs);
            }

            // Filter pairs that are already in the same cluster
            stdout.call_method1("write", ("Filtering processed pairs...\n".to_string(),))?;
            let filtered_pairs: Vec<(usize, usize)> = all_pairs.into_iter()
                .filter(|(i, j)| {
                    let root1 = self.find_cluster(*i);
                    let root2 = self.find_cluster(*j);
                    root1 != root2 && !self.is_processed(*i, *j)
                })
                .collect();

            stdout.call_method1("write", 
                (format!("Final number of pairs to process: {}\n", filtered_pairs.len()),))?;
            stdout.call_method0("flush")?;

            Ok(filtered_pairs)
        })
    }

    fn find_cluster(&mut self, mut item: usize) -> usize {
        while self.parent[item] != item {
            // Path compression: Point to grandparent to flatten tree
            self.parent[item] = self.parent[self.parent[item]];
            item = self.parent[item];
        }
        item
    }

    fn merge_clusters(&mut self, item1: usize, item2: usize) -> PyResult<()> {
        if item1 >= self.parent.len() || item2 >= self.parent.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Invalid cluster index"
            ));
        }
        let mut root1 = self.find_cluster(item1);
        let mut root2 = self.find_cluster(item2);
        
        if root1 != root2 {
            // Union by size - attach smaller tree to root of larger tree
            if self.size[root1] < self.size[root2] {
                std::mem::swap(&mut root1, &mut root2);
            }
            
            // Merge root2 into root1
            self.parent[root2] = root1;
            self.size[root1] += self.size[root2];
            
            // Merge clusters
            let items = self.clusters[root2].drain().collect::<Vec<_>>();
            self.clusters[root1].extend(items);
        }
        
        Ok(())
    }

    fn get_clusters(&self) -> PyResult<Vec<Vec<usize>>> {
        Ok(self.clusters.iter()
            .filter(|c| !c.is_empty())
            .map(|c| c.iter().copied().collect())
            .collect())
    }

    fn is_processed(&self, i: usize, j: usize) -> bool {
        let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
        self.processed_pairs.contains(&(min_idx, max_idx))
    }

    fn mark_processed(&mut self, i: usize, j: usize) {
        let (min_idx, max_idx) = if i < j { (i, j) } else { (j, i) };
        self.processed_pairs.insert((min_idx, max_idx));
    }
}

#[pymodule]
fn docetl_resolver(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FastResolver>()?;
    Ok(())
}