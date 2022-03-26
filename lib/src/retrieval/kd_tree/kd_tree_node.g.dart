// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'kd_tree_node.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KDTreeNode _$KDTreeNodeFromJson(Map<String, dynamic> json) {
  return $checkedNew('KDTreeNode', json, () {
    $checkKeys(json, allowedKeys: const ['V', 'I', 'L', 'R', 'P']);
    final val = KDTreeNode(
      value: $checkedConvert(json, 'V',
          (v) => v == null ? null : Vector.fromJson(v as Map<String, dynamic>)),
      splitIndex: $checkedConvert(json, 'I', (v) => v as int?),
      left: $checkedConvert(
          json,
          'L',
          (v) => v == null
              ? null
              : KDTreeNode.fromJson(v as Map<String, dynamic>)),
      right: $checkedConvert(
          json,
          'R',
          (v) => v == null
              ? null
              : KDTreeNode.fromJson(v as Map<String, dynamic>)),
      points: $checkedConvert(json, 'P',
          (v) => v == null ? null : Matrix.fromJson(v as Map<String, dynamic>)),
    );
    return val;
  }, fieldKeyMap: const {
    'value': 'V',
    'splitIndex': 'I',
    'left': 'L',
    'right': 'R',
    'points': 'P'
  });
}

Map<String, dynamic> _$KDTreeNodeToJson(KDTreeNode instance) {
  final val = <String, dynamic>{};

  void writeNotNull(String key, dynamic value) {
    if (value != null) {
      val[key] = value;
    }
  }

  writeNotNull('V', instance.value?.toJson());
  writeNotNull('I', instance.splitIndex);
  writeNotNull('L', instance.left?.toJson());
  writeNotNull('R', instance.right?.toJson());
  writeNotNull('P', instance.points?.toJson());
  return val;
}
