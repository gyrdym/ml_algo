// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'tree_node.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

TreeNode _$TreeNodeFromJson(Map<String, dynamic> json) {
  return $checkedNew('TreeNode', json, () {
    $checkKeys(json, allowedKeys: const ['CN', 'LB', 'PT', 'SV', 'SI', 'LV']);
    final val = TreeNode(
      $checkedConvert(
          json, 'PT', (v) => fromSplittingPredicateTypeJson(v as String)),
      $checkedConvert(json, 'SV', (v) => v as num),
      $checkedConvert(json, 'SI', (v) => v as int),
      $checkedConvert(json, 'CN',
          (v) => fromTreeNodesJson(v as Iterable<Map<String, dynamic>>)),
      $checkedConvert(
          json, 'LB', (v) => fromLeafLabelJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'LV', (v) => v as int),
    );
    return val;
  }, fieldKeyMap: const {
    'predicateType': 'PT',
    'splittingValue': 'SV',
    'splittingIndex': 'SI',
    'children': 'CN',
    'label': 'LB',
    'level': 'LV'
  });
}

Map<String, dynamic> _$TreeNodeToJson(TreeNode instance) => <String, dynamic>{
      'CN': treeNodesToJson(instance.children),
      'LB': leafLabelToJson(instance.label),
      'PT': splittingPredicateTypeToJson(instance.predicateType),
      'SV': instance.splittingValue,
      'SI': instance.splittingIndex,
      'LV': instance.level,
    };
