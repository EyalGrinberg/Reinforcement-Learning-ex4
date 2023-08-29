import random
import itertools

def deal(deck): # Deal two cards from the current deck
    hand = []
    for i in range(2):
        random.shuffle(deck)
        card = deck.pop()
        if card == 11:card = "J"
        if card == 12:card = "Q"
        if card == 13:card = "K"
        if card == 14:card = "A"
        hand.append(card)       
    return hand

def total(hand): # Compute the total sum of the current hand
    total = 0
    for card in hand:
        if card in ["J", "Q", "K"]:
            total += 10
        elif card == "A":
            total += 11
        else:
            total += card
    return total

def hit(deck): # Hit another card from the current deck
    card = deck.pop()
    if card == 11:
        card = "J"
    elif card == 12:
        card = "Q"
    elif card == 13:
        card = "K"
    elif card == 14:
        card = "A"
    return card

def reward(player_hand, dealer_hand): # Compute the reward - 1 if won, 0 otherwise
    player = total(player_hand)
    dealer = total(dealer_hand)
    if player == 21: # If the gambler got a blackjack, the gambler won
        return 1
    if player < 21 and dealer > 21: # If dealer burnt out, the gambler won
        return 1
    if player < 21 and dealer < 21 and dealer < player: # If neither burnt out, but gambler has higher sum, the gambler won
        return 1
    return 0

def game(p):
    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]*4 # Init. the deck
    player_hand = deal(deck) # Deal the gambler
    dealer_hand = deal(deck) # Deal the dealer
    sums = [] # Record the sums reached in the game
    sums.append(total(player_hand))
    while p(player_hand) == 1: # As long as policy wants to hit, continue 
        player_hand.append(hit(deck))
        sums.append(total(player_hand))
    while total(dealer_hand) < 16: # Naive policy of the dealer - keep hitting until hitting a sum greater than 16
        dealer_hand.append(hit(deck))
    r = reward(player_hand, dealer_hand) # Compute reward
    return r, sums # Return reward and sums during the game

def main():
    random.seed(42)
    l = len(range(4, 29))
    V = [0] * l
    N = 100000
    lr = 0.01  
    gamma = 1
    
    def p(player_hand): # Naive policy of the gambler - keep hitting until hitting a sum greater than 17
        if total(player_hand) < 18:
            return 1
        return 0
    
    for _ in range(N):
        r, sums = game(p) # Run an episode, tracking the term. reward and sums during the game
        for j in range(len(sums)-1): # Update V-function
            idx = sums[j] - 4 # s_{t} of the TD(0) update
            next_idx = sums[j+1]-4 # s_{t+1} of the TD(0) update
            # The reward for intermediate states is 0
            # V_{t+1}(s_{t}) = V_{t}(s_{t}) + lr * (r_{t} + gamma * V_{t}(s_{t+1}) - V_{t}(s_{t}))
            V[idx] = V[idx] + lr * (0 + gamma*V[next_idx] - V[idx])
        # The final state has a value of 0, and the final reward is r
        V[sums[-1]-4] = V[sums[-1]-4] + lr * (r + gamma*0 - V[sums[-1]-4])
    
    for i in range(l):
        print(f'score: {range(4, 29)[i]}, prob: {V[i]}')

    # Code for computing probabilities to begin in each of the states
    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]*4
    combs = list(itertools.combinations(deck, 2))
    d = dict()
    for comb in combs:
        s = sum(comb)
        if s not in d:
            d[s] = 1
        else:
            d[s] += 1
    for key in d.keys():
        d[key] /= 1326
        #print(f'key:{key}, prob:{d[key]}')
    # Weighted average for the surrogate estimation of the prob. to win
    print(sum(d[idx+4] * V[idx] for idx in range(19)))

if __name__ == "__main__":
    main()