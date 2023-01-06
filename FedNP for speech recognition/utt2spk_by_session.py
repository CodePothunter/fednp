#!/usr/bin/env python
# coding=utf-8
import sys
import os

num_args = len(sys.argv)
if sys.argv[1] == "4": # means seperated by four parties
    sess2party = {"S03":"PT1", "S04":"PT1", "S05":"PT2", "S06":"PT3", "S07":"PT3", "S17":"PT3", "S08":"PT1", "S16":"PT1", "S12":"PT2", "S13":"PT4", "S18":"PT4", "S22":"PT4", "S19":"PT2", "S20":"PT4", "S23":"PT2", "S24":"PT2"}
    root_utt2spk = sys.argv[2]
    dest_utt2spks = sys.argv[3:]
elif sys.argv[1] == '8': # separated by similar speaker
    sess2party = {"S03":"PT1", "S04":"PT1", "S05":"PT2", "S06":"PT2", "S07":"PT3", "S17":"PT3", "S08":"PT4", "S16":"PT4", "S12":"PT5", "S13":"PT5", "S18":"PT6", "S22":"PT6", "S19":"PT7", "S20":"PT7", "S23":"PT8", "S24":"PT8"} 
    root_utt2spk = sys.argv[2]
    dest_utt2spks = sys.argv[3:]
else:
    sess2party = {"S03":"S03", "S04":"S04", "S05":"S05", "S06":"S06", "S07":"S07", "S17":"S17", "S08":"S08", "S16":"S16", "S12":"S12", "S13":"S13", "S18":"S18", "S22":"S22", "S19":"S19", "S20":"S20", "S23":"S23", "S24":"S24"}
    root_utt2spk = sys.argv[1]
    dest_utt2spks = sys.argv[2:]

assert os.path.exists(root_utt2spk), "root utt2spk file doesnot exists"

session2lines = {}
with open(root_utt2spk) as f:
    for line in f:
        spk, sess, _ = line.split("_")
        if sess2party[sess] not in session2lines:
            session2lines[sess2party[sess]] = []
        session2lines[sess2party[sess]].append(line)

assert len(dest_utt2spks) == len(session2lines), "number of splits not matches"

for dest_file, party in zip(dest_utt2spks, session2lines.keys()):
    with open(dest_file, "w") as f:
        for line in session2lines[party]:
            f.write(line)

print("Done")
exit(0)



