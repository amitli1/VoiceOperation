from StatusParser.main_status_parser import ask_llm


class StatusTesterManager:
    def __init__(self):
        pass

    def create_power_status_message(self):
        power_source_status_msg = [
            {
                "acPower": 0,
                "vehiclePower": 0,
                "version": "2.0",
                "defaultTopicName": "PowerSourceData",
                "batteryStatus": {
                    "charging": 0,
                    "voltageLevel": 15
                },
                "insPower": 0,
                "gfePower": 0,
                "lcuPower": 9,
                "mcuPower": 2
            }
        ]
        return power_source_status_msg

    def create_test_version(self):
        user_q     = "What is LMU power version ?"
        answer     = "2.0"
        return user_q,  answer

    def create_test_charging(self):
        user_q = "Is the battery in charging mode ?"
        answer = "no"
        return user_q, answer

    def create_test_lcu_and_mcu_power(self):
        user_q = "What is the LCU and MCU battery power ?"
        answer = "9, 2"
        return user_q, answer

    def create_test_battery_voltage(self):
        user_q = "What is battery voltage ?"
        answer = "15"
        return user_q, answer



    def run_tests(self, model_name, schema):

        power_source_status_msg = self.create_power_status_message()

        user_q, gt_answer       = self.create_test_version()
        results                 = ask_llm(model_name, user_q, power_source_status_msg, schema)
        print(f"Q: {user_q}")
        print(f"\tAnswer : {results}")
        print(f"\tGT     : {gt_answer}")

        user_q, gt_answer = self.create_test_charging()
        results           = ask_llm(model_name, user_q, power_source_status_msg, schema)
        print(f"Q: {user_q}")
        print(f"\tAnswer : {results}")
        print(f"\tGT     : {gt_answer}")

        user_q, gt_answer = self.create_test_lcu_and_mcu_power()
        results = ask_llm(model_name, user_q, power_source_status_msg, schema)
        print(f"Q: {user_q}")
        print(f"\tAnswer : {results}")
        print(f"\tGT     : {gt_answer}")

        user_q, gt_answer = self.create_test_battery_voltage()
        results = ask_llm(model_name, user_q, power_source_status_msg, schema)
        print(f"Q: {user_q}")
        print(f"\tAnswer : {results}")
        print(f"\tGT     : {gt_answer}")

