class Neighbour {
  Neighbour(this.index, this.distance);

  final int index;
  final num distance;

  @override
  bool operator ==(Object other) {
    if (other is Neighbour) {
      return index == other.index;
    }

    return false;
  }

  @override
  int get hashCode => '$index:$distance'.hashCode;

  @override
  String toString() => '(Index: $index, Distance: $distance)';
}
