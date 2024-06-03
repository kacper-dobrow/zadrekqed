import requests
import json


def send_request(data):
    url = "http://127.0.0.1:5000/predict"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        print(f"Success:")
        print(response.json())
    else:
        print("Error:")
        print(response.status_code)
        print(response.json())


if __name__ == '__main__':
    # Valid request
    valid_data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    print("Sending valid request...")
    send_request(valid_data)

    # Invalid request (insufficient dimensions)
    invalid_data_1 = [
        [1.0, 2.0],
        [4.0, 5.0]
    ]
    print("\nSending invalid request (insufficient dimensions)...")
    send_request(invalid_data_1)

    # Invalid request (invalid data types)
    invalid_data_2 = [
        [1.0, "2.0", 3.0],
        [4.0, 5.0, "6.0"]
    ]
    print("\nSending invalid request (invalid data types)...")
    send_request(invalid_data_2)