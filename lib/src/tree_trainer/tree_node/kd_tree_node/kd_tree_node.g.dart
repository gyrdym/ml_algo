// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'kd_tree_node.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KDTreeNode _$KDTreeNodeFromJson(Map<String, dynamic> json) {
  return $checkedNew('KDTreeNode', json, () {
    $checkKeys(json, allowedKeys: const ['C', 'S', 'P', 'V', 'I', 'L']);
    final val = KDTreeNode(
      $checkedConvert(
          json, 'P', (v) => fromSplittingPredicateTypeJson(v as String?)),
      $checkedConvert(json, 'V', (v) => v as num?),
      $checkedConvert(json, 'I', (v) => v as int?),
      $checkedConvert(json, 'C', (v) => fromKDTreeNodesJson(v as List?)),
      $checkedConvert(json, 'S',
          (v) => v == null ? null : Matrix.fromJson(v as Map<String, dynamic>)),
      $checkedConvert(json, 'L', (v) => v as int),
    );
    return val;
  }, fieldKeyMap: const {
    'predicateType': 'P',
    'splittingValue': 'V',
    'splittingIndex': 'I',
    'children': 'C',
    'samples': 'S',
    'level': 'L'
  });
}

Map<String, dynamic> _$KDTreeNodeToJson(KDTreeNode instance) {
  final val = <String, dynamic>{};

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('C', kdTreeNodesToJson(instance.children));
  writeNotNull('S', instance.samples?.toJson());
  writeNotNull('P', splittingPredicateTypeToJson(instance.predicateType));
  writeNotNull('V', instance.splittingValue);
  writeNotNull('I', instance.splittingIndex);
  val['L'] = instance.level;
  return val;
}
