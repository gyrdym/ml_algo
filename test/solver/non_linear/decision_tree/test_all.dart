import 'assessor/majority_split_assesor_test.dart' as majority_split_assessor_test;
import 'decision_tree_solver_integration_test.dart' as decision_tree_solver_integration_test;
import 'leaf_detector/leaf_detector_impl_test.dart' as leaf_detector_impl_test;
import 'leaf_label_factory/majority_leaf_label_factory_test.dart' as majority_leaf_label_factory_test;
import 'split_selector/greedy_split_selector_test.dart' as greedy_split_selector_test;
import 'splitter/greedy_splitter_test.dart' as greedy_splitter_test;
import 'splitter/nominal_splitter/nominal_splitter_impl_test.dart' as nominal_splitter_impl_test;
import 'splitter/numerical_splitter/numerical_splitter_impl_test.dart' as numerical_splitter_impl_test;

void main() {
  majority_split_assessor_test.main();
  leaf_detector_impl_test.main();
  majority_leaf_label_factory_test.main();
  numerical_splitter_impl_test.main();
  nominal_splitter_impl_test.main();
  greedy_splitter_test.main();
  greedy_split_selector_test.main();
  decision_tree_solver_integration_test.main();
}
