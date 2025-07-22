import zmq
import json

def send_command_to_queue_server(command, params=None):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect("tcp://localhost:60615")

    msg = {"method": command}
    if params:
        msg["params"] = params

    socket.send_json(msg)
    reply = socket.recv_json()
    return reply

# ✅ Check connection
print("RE Manager Status:")
print(send_command_to_queue_server("status"))

# ✅ Add a plan
plan = {
    "item_type": "plan",
    "name": "count",
    "args": [["det"]],
    "kwargs": {}
}

params = {
    "item": plan,
    "user": "test_user",          # Required
    "user_group": "primary"       # Default allowed group
}

print("Adding a plan:")
print(send_command_to_queue_server("queue_item_add", params))
 
# import zmq
# import json

# def send_command_to_queue_server(command, params=None):
#     ctx = zmq.Context()
#     socket = ctx.socket(zmq.REQ)
#     socket.connect("tcp://localhost:60615")

#     msg = {"method": command}
#     if params:
#         msg["params"] = params

#     socket.send_json(msg)
#     reply = socket.recv_json()
#     return reply

# # Define the plan arguments
# plan_name = "recover_pos_and_scan"
# label = "coarse_scan"
# roi = [1,2,3]      # Your ROI dictionary or data structure
# dets = "Dets"     # List of detectors, e.g., ["det1", "det2"]
# x_motor = "motor_x"
# x_start = 0
# x_end = 10
# mot1_n = 5
# y_motor = "motor_y"
# y_start = 0
# y_end = 10
# mot2_n = 5

# plan = {
#     "item_type": "plan",
#     "name": plan_name,
#     "args": [
#         label,
#         roi,
#         dets,
#         x_motor,
#         x_start,
#         x_end,
#         mot1_n,
#         y_motor,
#         y_start,
#         y_end,
#         mot2_n
#     ],
#     "kwargs": {}
# }

# params = {
#     "item": plan,
#     "user": "test_user",
#     "user_group": "primary"
# }

# print("Adding a recover_pos_and_scan plan:")
# response = send_command_to_queue_server("queue_item_add", params)
# print(response)
