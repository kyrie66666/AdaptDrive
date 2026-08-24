#!/bin/bash
pkill -9 -f 'CarlaUE4-Linux-Shipping' > /dev/null 2>&1 || true
pkill -9 -f 'carla-rpc-port=' > /dev/null 2>&1 || true
pkill -9 -f 'leaderboard_evaluator.py' > /dev/null 2>&1 || true
