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
          json,
          'PT',
          (v) =>
              _$enumDecodeNullable(_$TreeNodeSplittingPredicateTypeEnumMap, v)),
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
      'PT': _$TreeNodeSplittingPredicateTypeEnumMap[instance.predicateType],
      'SV': instance.splittingValue,
      'SI': instance.splittingIndex,
      'LV': instance.level,
    };

T _$enumDecode<T>(
  Map<T, dynamic> enumValues,
  dynamic source, {
  T unknownValue,
}) {
  if (source == null) {
    throw ArgumentError('A value must be provided. Supported values: '
        '${enumValues.values.join(', ')}');
  }

  final value = enumValues.entries
      .singleWhere((e) => e.value == source, orElse: () => null)
      ?.key;

  if (value == null && unknownValue == null) {
    throw ArgumentError('`$source` is not one of the supported values: '
        '${enumValues.values.join(', ')}');
  }
  return value ?? unknownValue;
}

T _$enumDecodeNullable<T>(
  Map<T, dynamic> enumValues,
  dynamic source, {
  T unknownValue,
}) {
  if (source == null) {
    return null;
  }
  return _$enumDecode<T>(enumValues, source, unknownValue: unknownValue);
}

const _$TreeNodeSplittingPredicateTypeEnumMap = {
  TreeNodeSplittingPredicateType.lessThan: 'lessThan',
  TreeNodeSplittingPredicateType.lessThanOrEqualTo: 'lessThanOrEqualTo',
  TreeNodeSplittingPredicateType.equalTo: 'equalTo',
  TreeNodeSplittingPredicateType.greaterThanOrEqualTo: 'greaterThanOrEqualTo',
  TreeNodeSplittingPredicateType.greaterThan: 'greaterThan',
};
