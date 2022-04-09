import 'package:ml_algo/src/tree_trainer/tree_node/tree_node_json_keys.dart';

Map<String, dynamic> migrateTreeNodeJsonSchema(Map<String, dynamic> json) {
  if (json.containsKey(levelJsonKey)) {
    json.remove(levelJsonKey);
  }

  return json;
}
