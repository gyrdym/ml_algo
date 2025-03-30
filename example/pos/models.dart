// ignore_for_file: public_member_api_docs, sort_constructors_first

import 'dart:convert';

class Item {
  final double articleId;
  final double qt;

  Item({required this.articleId, required this.qt});

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'articleId': articleId,
      'qt': qt,
    };
  }

  factory Item.fromMap(Map<String, dynamic> map) {
    return Item(
      articleId: double.parse('${map['articleId']}'),
      qt: double.parse('${map['qt']}'),
    );
  }
  String toJson() => json.encode(toMap());

  factory Item.fromJson(String source) =>
      Item.fromMap(json.decode(source) as Map<String, dynamic>);

  @override
  String toString() => '$articleId * $qt';
}

class Ticket {
  final List<Item> items;
  final double contactId;
  final int timestamp;

  Ticket(
      {required this.items, required this.contactId, required this.timestamp});

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'timestamp': timestamp,
      'contactId': contactId,
      'items': items.map((x) => x.toMap()).toList(),
    };
  }

  factory Ticket.fromMap(Map<String, dynamic> map) {
    return Ticket(
      items: List<Item>.from(
        (map['items'] as List<dynamic>).map<Item>(
          (x) => Item.fromMap(x as Map<String, dynamic>),
        ),
      ),
      contactId: double.parse('${map['contactId']}') ,
      timestamp: int.tryParse(map['timestamp']['\$numberLong'] as String) ?? 0,
    );
  }
  String toJson() => json.encode(toMap());
  factory Ticket.fromJson(String source) =>
      Ticket.fromMap(json.decode(source) as Map<String, dynamic>);

  @override
  String toString() => '$timestamp, $contactId, $items';
}

class TicketWrapper {
  final Ticket ticket;
  TicketWrapper({required this.ticket});

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'ticket': ticket.toMap(),
    };
  }

  factory TicketWrapper.fromMap(Map<String, dynamic> map) {
    return TicketWrapper(
      ticket: Ticket.fromMap(map['ticket'] as Map<String, dynamic>),
    );
  }
  String toJson() => json.encode(toMap());
  factory TicketWrapper.fromJson(String source) =>
      TicketWrapper.fromMap(json.decode(source) as Map<String, dynamic>);
}


const availableContactIds = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    55,
    57
  ];