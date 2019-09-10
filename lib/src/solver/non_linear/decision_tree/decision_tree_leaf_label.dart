class DecisionTreeLeafLabel {
  DecisionTreeLeafLabel(this.value, {this.probability});

  final double value;
  final double probability;

  Map<String, dynamic> serialize() => <String, dynamic>{
    'value': value,
    'probability': probability,
  };
}
