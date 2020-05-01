from Lang import *
from utils import *
from Decoder import  *
from Encoder import  *
from packages import *
from data_processing import *

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):

        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def bi_directional_beam_decode(decoder_forwards, decoder_backwards):

    beam_width = 10
    topk = 3
    decoded_batch = []

    for idx in range(MAX_LENGTH):
        if isinstance(decoder_forwards, tuple): 
            decoder_hidden = (decoder_forwards[0][:,idx, :].unsqueeze(0),decoder_forwards[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_forwards[:, idx, :].unsqueeze(0)
        decoder_backwards = encoder_outputs[:,idx, :].unsqueeze(1)
        decoder_input = torch.LongTensor([[SOS_token]], device=device)
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        nodes.put((-node.eval(), node))
        qsize = 1

        while True:
            if qsize > 2000: break
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, decoder_backwards)
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
            qsize += len(nextnodes) - 1

        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch
