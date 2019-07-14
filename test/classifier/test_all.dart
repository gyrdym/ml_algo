import 'decision_tree/decision_tree_classifier_impl_test.dart' as decision_tree_classifier_impl_test;
import 'decision_tree/decision_tree_classifier_test.dart' as decision_tree_classifier_test;
import 'logistic_regressor/logistic_regressor_integration_test.dart' as logistic_regressor_integration_test;
import 'logistic_regressor/logistic_regressor_test.dart' as logistic_regressor_test;
import 'softmax_regressor/softmax_regressor_test.dart' as softmax_regressor_test;

void main() {
  decision_tree_classifier_impl_test.main();
  decision_tree_classifier_test.main();
  logistic_regressor_integration_test.main();
  logistic_regressor_test.main();
  softmax_regressor_test.main();
}
