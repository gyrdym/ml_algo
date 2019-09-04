class DecisionTreeLeafLabel {
  DecisionTreeLeafLabel.nominal(this.nominalValue, {this.probability})
      : numericalValue = null;

  DecisionTreeLeafLabel.numerical(this.numericalValue, {this.probability})
      : nominalValue = null;

  final dynamic nominalValue;
  final double numericalValue;
  final double probability;
}
