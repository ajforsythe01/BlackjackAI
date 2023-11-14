import os
import random
from enum import Enum


class Blackjack:
    class State(Enum):
        WIN = 1
        LOSE = 2
        TIE = 3
        PLAYING = 4
    class Turn(Enum):
        PLAYER = 1
        DEALER = 2
    class Hand:
        def __init__(self, deck):
            self.hand = []
            self.soft = False
            for i in range(2):
                card = self.convert(deck.pop())
                self.hand.append(card)
            
        def hit(self, deck):
            card = self.convert(deck.pop())
            self.hand.append(card)
        
        def convert(self, card):
            if card == 11:card = "J"
            if card == 12:card = "Q"
            if card == 13:card = "K"
            if card == 14:card = "A"
            return card
         
        def value(self, card):
            if card == "J" or card == "Q" or card == "K":
                card = 10
            elif card == 'A':
                card = 11
            return card
            
        def total(self):
            total = 0
            num_aces = 0
            
            for card in self.hand:
                if card == "J" or card == "Q" or card == "K":
                    total += 10
                elif card == "A":
                    num_aces += 1
                else:
                    total += card

            # Calculate the total with Ace value consideration
            self.soft = False
            for _ in range(num_aces):
                if total + 11 <= 21:
                    total += 11
                    self.soft = True
                else:
                    total += 1
                    
            return total
        
        def difference21(self):
            return 21 - self.total()
        
        def check21(self):
            if self.total() == 21:
                return True         
            return False
        
        def has_ace(self):
            if len(self.hand) == 2:
                return "A" in self.hand
            return False

        def is_soft(self):
            return self.soft
            
        def has_pair(self):
            if len(self.hand) == 2:
                ranks = []
                for card in self.hand:
                    if card == "10" or card in ["J", "Q", "K"]:
                        ranks.append("10JQK")
                    else:
                        ranks.append(card)
                return ranks[0] == ranks[1]
            return False
            
    def __init__(self):
        self.decks = 6
        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]*(self.decks*4)
        random.shuffle(self.deck)
        self.player = self.Hand(self.deck)
        self.dealer = self.Hand(self.deck)
        if self.player.check21() and self.dealer.check21():
            self.state = self.State.TIE
        elif self.player.check21():
            self.state = self.State.WIN
        elif self.dealer.check21():
            self.state = self.State.LOSE
        else:
            self.state = self.State.PLAYING
            self.turn = self.Turn.PLAYER
       
    def stand(self):
        while self.dealer.total() < 17:
            self.dealer.hit(self.deck)
            # Check for Dealer bust
            if self.dealer.total() > 21:
                self.state = self.State.WIN
        
        if self.state == self.State.PLAYING:
            self.determinewinner()
    
    def hit(self):
        # This handles player hitting. Dealer hits are made in the stand function.
        if self.state == self.State.PLAYING and self.turn == self.Turn.PLAYER:
            self.player.hit(self.deck)
            if self.player.total() == 21:
                self.stand()
        if self.player.total() > 21:
            self.state = self.State.LOSE
    
    def determinewinner(self):
        if self.player.difference21() > self.dealer.difference21():
            self.state = self.State.LOSE
        elif self.player.difference21() < self.dealer.difference21():
            self.state = self.State.WIN
        else:
            self.state = self.State.TIE
        
if __name__ == "__main__":
    while True:
        bj = Blackjack()
        print("Welcome to Blackjack!")
        print(f"Dealer's upcard: {bj.dealer.hand[0]}")
        print(f"Player's hand: {bj.player.hand} (Total: {bj.player.total()})")
        print()

        # Check for dealer's blackjack
        if bj.dealer.check21():
            print(f"Dealer has a blackjack! (Total: {bj.dealer.total()})")
            if bj.player.check21():
                bj.state = bj.State.TIE
            else:
                bj.state = bj.State.LOSE

        # Player's turn
        while bj.state == bj.State.PLAYING and bj.turn == bj.Turn.PLAYER:
            action = input("Do you want to hit or stand? ").strip().lower()
            if action == 'hit':
                bj.hit()
                print(f"Player's hand: {bj.player.hand} (Total: {bj.player.total()})")
                print()
                if bj.player.total() > 21:
                    bj.state = bj.State.LOSE
                    break
            elif action == 'stand':
                bj.turn = bj.Turn.DEALER
                print()

        # Dealer's turn
        if bj.state == bj.State.PLAYING:
            bj.stand()
            print(f"Dealer's hand: {bj.dealer.hand} (Total: {bj.dealer.total()})")
            print()

        # Determine the winner
        if bj.state == bj.State.TIE:
            print("It's a tie!")
        elif bj.state == bj.State.WIN:
            print("Player wins!")
        else:
            print("Dealer wins!")

        # Ask the player if they want to play another round
        play_again = input("Do you want to play another round? (yes/no): ").strip().lower()
        if play_again != 'yes':
            break