// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'tree_node.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

TreeNode _$TreeNodeFromJson(Map<String, dynamic> json) {
  return $checkedNew('TreeNode', json, () {
    $checkKeys(json, allowedKeys: const ['CN', 'LB', 'PT', 'SV', 'SI', 'LV']);
    final val = TreeNode(
      $checkedConvert(json, 'PT', (v) => fromPredicateTypeJson(v as String?)),
      $checkedConvert(json, 'SV', (v) => v as num?),
      $checkedConvert(json, 'SI', (v) => v as int?),
      $checkedConvert(json, 'CN', (v) => fromTreeNodesJson(v as List?)),
      $checkedConvert(
          json,
          'LB',
          (v) => v == null
              ? null
              : TreeLeafLabel.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'LV', (v) => v as int),
    );
    return val;
  }, fieldKeyMap: const {
    'predicateType': 'PT',
    'splitValue': 'SV',
    'splitIndex': 'SI',
    'children': 'CN',
    'label': 'LB',
    'level': 'LV'
  });
}

Map<String, dynamic> _$TreeNodeToJson(TreeNode instance) {
  final val = <String, dynamic>{};

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('CN', treeNodesToJson(instance.children));
  writeNotNull('LB', instance.label?.toJson());
  writeNotNull('PT', predicateTypeToJson(instance.predicateType));
  writeNotNull('SV', instance.splitValue);
  writeNotNull('SI', instance.splitIndex);
  val['LV'] = instance.level;
  return val;
}
