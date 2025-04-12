# Main app script with all features: QR scan, clustering, route map, final delivery log
# (Same imports as before, omitted here for brevity)

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.garden.zbarcam import ZBarCam




import openrouteservice
import csv
import os

import requests
import webbrowser

from sklearn.cluster import KMeans
import folium
import matplotlib.pyplot as plt
from folium.features import CustomIcon

client = None


class QRScanner(BoxLayout):
    def __init__(self, **kwargs):
        super(QRScanner, self).__init__(orientation='vertical', **kwargs)

        self.label = Label(text="Click 'Scan QR' to begin", size_hint_y=None, height=40)
        self.scan_btn = Button(text="Scan QR Code", size_hint_y=None, height=40)
        self.scan_btn.bind(on_press=self.start_scan)

        self.save_btn = Button(text="Save to Log", size_hint_y=None, height=40)
        self.save_btn.bind(on_press=self.save_to_log)
        self.save_btn.disabled = True

        self.view_btn = Button(text="View/Edit Log", size_hint_y=None, height=40)
        self.view_btn.bind(on_press=self.view_log_popup)

        self.kmeans_btn = Button(text=" View KMeans Clustering", size_hint_y=None, height=40)
        self.kmeans_btn.bind(on_press=self.show_kmeans_plot)

        self.route_btn = Button(text=" Generate Truck Routes (Map)", size_hint_y=None, height=40)
        self.route_btn.bind(on_press=self.generate_routes_map)

        self.final_log_btn = Button(text="View Final Delivery Log", size_hint_y=None, height=40)
        self.final_log_btn.bind(on_press=self.view_final_delivery_log)

        self.ip_input = TextInput(hint_text="Enter PC IP (e.g. 192.168.1.100)", size_hint_y=None, height=40)
        self.send_btn = Button(text="Send Log to PC", size_hint_y=None, height=40)
        self.send_btn.bind(on_press=self.send_logs_to_pc)

        self.add_widget(self.label)
        self.add_widget(self.scan_btn)
        self.add_widget(self.save_btn)
        self.add_widget(self.view_btn)
        self.add_widget(self.kmeans_btn)
        self.add_widget(self.route_btn)
        self.add_widget(self.final_log_btn)
        self.add_widget(self.ip_input)
        self.add_widget(self.send_btn)

        self.parsed_data = {}
        self.log = []
        Clock.schedule_once(lambda dt: self.ask_api_key())

    def ask_api_key(self):
        content = BoxLayout(orientation='vertical')
        input_field = TextInput(hint_text='Enter OpenRouteService API Key')
        confirm_btn = Button(text='Set API Key')

        content.add_widget(input_field)
        content.add_widget(confirm_btn)

        popup = Popup(title='Enter API Key', content=content, size_hint=(0.9, 0.4))

        def set_key(instance):
            global client
            key = input_field.text.strip()
            if key:
                try:
                    client = openrouteservice.Client(key=key)
                    self.label.text = "API Key Set. You may now scan."
                    popup.dismiss()
                except Exception as e:
                    self.label.text = f"Invalid API Key: {e}"
            else:
                self.label.text = "API Key cannot be empty."

        confirm_btn.bind(on_press=set_key)
        popup.open()

    def start_scan(self, instance):
        self.popup = Popup(title="Scan QR Code")
        self.zbar = ZBarCam()
        self.zbar.bind(symbols=self.on_qr_detected)
        self.popup.content = self.zbar
        self.popup.open()



    def on_qr_detected(self, instance, symbols):
        if symbols:
            qr_data = symbols[0].data.decode("utf-8")
            self.label.text = f"Scanned:\n{qr_data}"
            self.parse_data(qr_data)
            self.save_btn.disabled = False
            self.popup.dismiss()

    def parse_data(self, text):
        required_fields = ['Fragility', 'Box_Dimension', 'Rotation_Lock', 'Weight', 'Address']
        missing = [field for field in required_fields if field not in parsed]
        if missing:
            self.label.text = f"❌ Missing fields in QR: {', '.join(missing)}"
            return

        parsed = {}
        if 'Stack_Lock' in parsed:
            parsed['Rotation_Lock'] = parsed.pop('Stack_Lock')

        for item in text.split(';'):
            if '=' in item:
                key, value = item.strip().split('=', 1)
                parsed[key.strip()] = value.strip()
        address = parsed.get('Address', '')
        lat, lon = self.get_coordinates(address)
        parsed['Latitude'] = lat
        parsed['Longitude'] = lon
        self.parsed_data = parsed
        if lat is None or lon is None:
            self.label.text = "⚠️ Address geocoding failed. Lat/Lon not available!"

    def get_coordinates(self, address):
        global client
        try:
            response = client.pelias_search(text=address)
            features = response.get('features', [])
            if features:
                coords = features[0]['geometry']['coordinates']
                return coords[1], coords[0]
        except Exception as e:
            print(f"Geocode error: {e}")
        return None, None

    def save_to_log(self, instance):
        self.log.append(self.parsed_data.copy())
        self.label.text = "Package added to log."
        self.save_btn.disabled = True

    def view_log_popup(self, instance):
        content = BoxLayout(orientation='vertical', spacing=10)
        scroll = ScrollView(size_hint=(1, 0.8))
        grid = GridLayout(cols=1, spacing=5, size_hint_y=None)
        grid.bind(minimum_height=grid.setter('height'))

        for idx, entry in enumerate(self.log):
            summary = f"{entry.get('Box_Dimension', '?')} | {entry.get('Weight', '?')} | {entry.get('Fragility', '?')} | {entry.get('Rotation_Lock', '?')} | {entry.get('Latitude', '?')}, {entry.get('Longitude', '?')} | {entry.get('Address', '')}"
            line = BoxLayout(size_hint_y=None, height=30)
            line.add_widget(Label(text=summary))
            remove_btn = Button(text="Delete", size_hint_x=None, width=80)
            remove_btn.bind(on_press=lambda btn, i=idx: self.remove_from_log(i))
            line.add_widget(remove_btn)
            grid.add_widget(line)

        scroll.add_widget(grid)
        close_btn = Button(text="Close", size_hint_y=None, height=40)
        content.add_widget(scroll)
        content.add_widget(close_btn)

        popup = Popup(title="Logged Packages", content=content, size_hint=(0.9, 0.95))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

    def remove_from_log(self, index):
        if 0 <= index < len(self.log):
            del self.log[index]
            self.label.text = "Package removed from log."

    def send_logs_to_pc(self, instance):
        ip = self.ip_input.text.strip()
        if not ip:
            self.label.text = "Please enter PC IP address."
            return
        csv_path = os.path.join(App.get_running_app().user_data_dir, 'scanned_qr_data.csv')
        fields = ['Fragility', 'Box_Dimension', 'Rotation_Lock', 'Weight', 'Address', 'Latitude', 'Longitude']
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for item in self.log:
                writer.writerow(item)
        try:
            with open(csv_path, 'rb') as f:
                files = {'file': ('scanned_qr_data.csv', f, 'text/csv')}
                url = f"http://{ip}:5001/upload"
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    self.label.text = "Logs sent to PC successfully."
                else:
                    self.label.text = f"Failed to send logs: {response.status_code}"
        except Exception as e:
            self.label.text = f"Error: {e}"

    def show_kmeans_plot(self, instance):
        coords = [(float(p['Latitude']), float(p['Longitude'])) for p in self.log if p.get('Latitude') and p.get('Longitude')]
        if len(coords) < 3:
            self.label.text = "Need at least 3 GPS points."
            return
        kmeans = KMeans(n_clusters=3, random_state=0).fit(coords)
        labels = kmeans.labels_
        lat = [p[0] for p in coords]
        lon = [p[1] for p in coords]
        plt.figure(figsize=(8, 6))
        plt.scatter(lon, lat, c=labels, cmap='rainbow', s=100, edgecolors='k')
        plt.title("KMeans Zone Clustering (3 Trucks)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.savefig('kmeans_plot.png')
        webbrowser.open('kmeans_plot.png')  # works only if you move it to user_data_dir

    def generate_routes_map(self, instance):
        coords = [(float(p['Latitude']), float(p['Longitude'])) for p in self.log if p.get('Latitude') and p.get('Longitude')]
        if len(coords) < 3:
            self.label.text = "Need at least 3 GPS points."
            return
        kmeans = KMeans(n_clusters=3, random_state=0).fit(coords)
        labels = kmeans.labels_
        zones = {0: [], 1: [], 2: []}
        for i, label in enumerate(labels):
            zones[label].append([coords[i][1], coords[i][0]])  # lon, lat

        fmap = folium.Map(location=coords[0], zoom_start=10)
        truck_icon_url = "https://cdn-icons-png.flaticon.com/512/1995/1995474.png"
        colors = ['red', 'green', 'blue']

        for zone_idx, zone_coords in zones.items():
            if len(zone_coords) >= 2:
                try:
                    route = client.directions(coordinates=zone_coords, profile='driving-car', format='geojson')
                    summary = route['features'][0]['properties']['summary']
                    duration = summary['duration'] / 60
                    distance = summary['distance'] / 1000
                    folium.GeoJson(route, name=f"Truck {zone_idx+1}").add_to(fmap)
                    label = f"Truck {zone_idx+1}\n{distance:.1f} km, {duration:.0f} min"
                    start = zone_coords[0]
                    icon = CustomIcon(truck_icon_url, icon_size=(30, 30))
                    folium.Marker(location=[start[1], start[0]], tooltip=label, icon=icon).add_to(fmap)
                except Exception as e:
                    print(f"Routing error for zone {zone_idx}: {e}")

        map_path = os.path.join(App.get_running_app().user_data_dir, 'truck_routes_map.html')
        fmap.save(map_path)
        webbrowser.open(map_path)
        self.label.text = "Map with 3 truck routes generated."

    def view_final_delivery_log(self, instance):
        coords = [(float(p['Latitude']), float(p['Longitude'])) for p in self.log if p.get('Latitude') and p.get('Longitude')]
        if len(coords) < 3:
            self.label.text = "Need at least 3 GPS points."
            return
        kmeans = KMeans(n_clusters=3, random_state=0).fit(coords)
        labels = kmeans.labels_
        zones = {0: [], 1: [], 2: []}
        for i, label in enumerate(labels):
            point = self.log[i].copy()
            point['coords'] = [float(point['Longitude']), float(point['Latitude'])]
            zones[label].append(point)

        content = BoxLayout(orientation='vertical')
        scroll = ScrollView(size_hint=(1, 0.9))
        layout = GridLayout(cols=1, spacing=10, size_hint_y=None)
        layout.bind(minimum_height=layout.setter('height'))

        for truck_idx, points in zones.items():
            if len(points) < 2:
                continue
            try:
                coord_list = [p['coords'] for p in points]
                route = client.directions(coordinates=coord_list, profile='driving-car', format='geojson')
                route_coords = route['features'][0]['geometry']['coordinates']
                ordered = []
                used = set()
                for rc in route_coords:
                    nearest = min(
                        (p for p in points if id(p) not in used),
                        key=lambda x: (x['coords'][0] - rc[0])**2 + (x['coords'][1] - rc[1])**2
                    )
                    used.add(id(nearest))
                    ordered.append(nearest)
                for i, entry in enumerate(ordered):
                    entry['Sequence'] = i + 1

                layout.add_widget(Label(text=f"Truck {truck_idx+1} Delivery Order:", bold=True, size_hint_y=None, height=30))
                for entry in ordered:
                    line = (
                        f"{entry['Sequence']}) Box: {entry.get('Box_Dimension', '?')}, "
                        f"Weight: {entry.get('Weight', '?')}, Fragile: {entry.get('Fragility', '?')}, "
                        f"Stack: {entry.get('Rotation_Lock', '?')}, Addr: {entry.get('Address', '')}"
                    )
                    layout.add_widget(Label(text=line, size_hint_y=None, height=26))
            except Exception as e:
                layout.add_widget(Label(text=f"Routing error for Truck {truck_idx+1}: {e}", size_hint_y=None, height=30))

        scroll.add_widget(layout)
        close_btn = Button(text="Close", size_hint_y=None, height=40)
        content.add_widget(scroll)
        content.add_widget(close_btn)
        popup = Popup(title="Final Delivery Log (3 Trucks)", content=content, size_hint=(0.95, 0.95))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

class QRApp(App):
    def build(self):
        return QRScanner()

if __name__ == '__main__':
    QRApp().run()