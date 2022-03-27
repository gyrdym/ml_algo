// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'kd_tree_node.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

KDTreeNode _$KDTreeNodeFromJson(Map<String, dynamic> json) {
  return $checkedNew('KDTreeNode', json, () {
    $checkKeys(json, allowedKeys: const ['I', 'L', 'R', 'P']);
    final val = KDTreeNode(
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
      points: $checkedConvert(
          json, 'P', (v) => Matrix.fromJson(v as Map<String, dynamic>)),
    );
    return val;
  }, fieldKeyMap: const {
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

  writeNotNull('I', instance.splitIndex);
  writeNotNull('L', instance.left?.toJson());
  writeNotNull('R', instance.right?.toJson());
  val['P'] = instance.points.toJson();
  return val;
}
